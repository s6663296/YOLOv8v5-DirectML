# Module/tensorrt_inference.py
"""
TensorRT 推理策略，提供高效能的 NVIDIA GPU 推理。
需要安裝: tensorrt, cuda-python
"""
import os
import time
import numpy as np
import cv2
import traceback
from Module.logger import logger
from Module.config import Config

# TensorRT 相關導入
TRT_AVAILABLE = False
cudart = None

try:
    import tensorrt as trt
    # cuda-python v13.x 使用新的 API 結構
    try:
        from cuda.bindings import runtime as cudart
        logger.info("使用 cuda-python v13.x API")
    except ImportError:
        # cuda-python v12.x 及更早版本
        try:
            from cuda import cudart
            logger.info("使用 cuda-python v12.x API")
        except ImportError:
            # 嘗試直接使用 cuda 模組
            import cuda
            cudart = cuda
            logger.info("使用 cuda 模組")
    TRT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TensorRT 或 cuda-python 未安裝，TensorRT 推理模式不可用: {e}")


class TensorRTStrategy:
    """TensorRT 的高效能推理策略。"""
    
    def __init__(self, app_instance, image_processor=None):
        self.app = app_instance
        self.image_processor = image_processor
        self.engine = None
        self.context = None
        self.logger = trt.Logger(trt.Logger.WARNING) if TRT_AVAILABLE else None
        
        # 輸入輸出相關
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.model_names = {}
        self.num_classes = 80
        
        # CUDA 記憶體
        self.d_input = None
        self.d_output = None
        self.h_output = None
        self.h_input = None  # Pinned host memory for input
        self.stream = None
        self.use_pinned_memory = True  # 使用固定記憶體加速傳輸
        self.use_high_priority_stream = True  # 使用高優先級 stream
        self._preprocess_resize_buffer = None
        self._preprocess_float_buffer = None
        self._preprocess_input_buffer = None
        self._input_scale = np.float32(1.0 / 255.0)
        
        # YOLO 版本 (用於後處理)
        self.yolo_version = getattr(app_instance, 'selected_yolo_version', 'v8')
    
    def initialize(self) -> bool:
        """載入 TensorRT Engine 並準備推理。"""
        if not TRT_AVAILABLE:
            logger.error("TensorRT 未安裝，無法使用 TensorRT 推理模式")
            return False
        
        model_path = Config.get("model_file")
        if not model_path:
            logger.error("在設定中未配置模型檔案路徑。")
            return False
        
        # 確認是 .engine 檔案
        if not model_path.endswith('.engine'):
            logger.error(f"TensorRT 模式需要 .engine 檔案，但提供的是: {model_path}")
            logger.error("請使用 trt_converter_gui.py 或 run_trt_converter_gui.bat 轉換為 TensorRT Engine")
            return False
        
        if not os.path.isabs(model_path):
            from Module.utils import resource_path
            model_path = resource_path(model_path)
        
        try:
            if not os.path.exists(model_path):
                logger.error(f"找不到 TensorRT Engine 檔案: {model_path}")
                return False
            
            logger.info(f"正在從以下路徑載入 TensorRT Engine: {model_path}")
            
            # 載入 Engine
            runtime = trt.Runtime(self.logger)
            with open(model_path, 'rb') as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.engine is None:
                logger.error("無法反序列化 TensorRT Engine")
                return False
            
            # 創建執行上下文
            self.context = self.engine.create_execution_context()
            if self.context is None:
                logger.error("無法創建 TensorRT 執行上下文")
                return False
            
            # 獲取輸入輸出資訊
            self._setup_io_bindings()
            
            # 創建高優先級 CUDA stream (優先於遊戲渲染)
            if self.use_high_priority_stream:
                # 獲取 stream 優先級範圍
                err, priority_low, priority_high = cudart.cudaDeviceGetStreamPriorityRange()
                if err == cudart.cudaError_t.cudaSuccess and priority_high < priority_low:
                    # 使用最高優先級
                    err, self.stream = cudart.cudaStreamCreateWithPriority(
                        cudart.cudaStreamNonBlocking, priority_high
                    )
                    logger.info(f"創建高優先級 CUDA stream (優先級: {priority_high})")
                else:
                    err, self.stream = cudart.cudaStreamCreate()
                    logger.info("創建標準 CUDA stream")
            else:
                err, self.stream = cudart.cudaStreamCreate()
            
            if err != cudart.cudaError_t.cudaSuccess:
                logger.error(f"創建 CUDA stream 失敗: {err}")
                return False
            
            # 分配 GPU 記憶體 (包括 pinned memory)
            self._allocate_buffers()
            
            # 預熱
            logger.info("正在預熱 TensorRT Engine...")
            dummy_input = np.zeros(self.input_shape, dtype=np.float32)
            for _ in range(5):
                self._infer(dummy_input)
            
            logger.info("TensorRT Engine 初始化成功")
            logger.info(f"輸入形狀: {self.input_shape}, 輸出形狀: {self.output_shape}")
            return True
            
        except Exception as e:
            logger.error(f"初始化 TensorRT Engine 失敗: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _setup_io_bindings(self):
        """設定輸入輸出綁定。"""
        # TensorRT 10.x API
        num_io = self.engine.num_io_tensors
        
        for i in range(num_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                # 處理動態形狀
                if -1 in shape:
                    # 獲取優化配置的有效範圍
                    profile_idx = 0
                    min_shape = self.engine.get_tensor_profile_shape(name, profile_idx)[0]  # min
                    opt_shape = self.engine.get_tensor_profile_shape(name, profile_idx)[1]  # opt
                    max_shape = self.engine.get_tensor_profile_shape(name, profile_idx)[2]  # max
                    
                    # 使用 app 中的 model_size，但確保在有效範圍內
                    model_size = self.app.model_size if hasattr(self.app, 'model_size') and self.app.model_size else opt_shape[2]
                    
                    # 確保尺寸在優化範圍內
                    min_size = min_shape[2]
                    max_size = max_shape[2]
                    if model_size < min_size:
                        logger.warning(f"模型尺寸 {model_size} 小於 Engine 最小值 {min_size}，調整為 {min_size}")
                        model_size = min_size
                    elif model_size > max_size:
                        logger.warning(f"模型尺寸 {model_size} 大於 Engine 最大值 {max_size}，調整為 {max_size}")
                        model_size = max_size
                    
                    self.input_shape = (1, 3, model_size, model_size)
                    self.context.set_input_shape(name, self.input_shape)
                    
                    # 同步更新 app 的 model_size
                    if hasattr(self.app, 'model_size'):
                        self.app.model_size = model_size
                    
                    logger.info(f"動態形狀範圍: {min_shape} ~ {max_shape}，使用: {self.input_shape}")
                else:
                    self.input_shape = tuple(shape)
                    if hasattr(self.app, 'model_size'):
                        self.app.model_size = shape[2]
            else:
                self.output_name = name
        
        # 設定輸入形狀後，獲取實際輸出形狀
        output_shape = self.context.get_tensor_shape(self.output_name)
        
        # 處理動態輸出形狀 (將 -1 替換為實際值)
        if -1 in output_shape:
            # 對於 YOLO，輸出 anchors 數量與輸入尺寸相關
            # 通常是 (input_size / 8)^2 + (input_size / 16)^2 + (input_size / 32)^2
            input_h, input_w = self.input_shape[2], self.input_shape[3]
            num_anchors = (input_h // 8) * (input_w // 8) + \
                          (input_h // 16) * (input_w // 16) + \
                          (input_h // 32) * (input_w // 32)
            
            # 替換 -1 為計算值
            output_shape = list(output_shape)
            for idx, dim in enumerate(output_shape):
                if dim == -1:
                    output_shape[idx] = num_anchors
            output_shape = tuple(output_shape)
            logger.info(f"動態輸出形狀計算為: {output_shape}")
        
        self.output_shape = output_shape
        
        # 根據輸出形狀確定類別數量
        if len(self.output_shape) == 3:
            if self.yolo_version == 'v5':
                # YOLOv5: [batch, anchors, 5 + num_classes]
                self.num_classes = max(1, self.output_shape[2] - 5)
            else:
                # YOLOv8/v11/v12: [batch, 4 + num_classes, anchors]
                self.num_classes = max(1, self.output_shape[1] - 4)
        
        logger.info(f"TensorRT 輸入: {self.input_name}, 形狀: {self.input_shape}")
        logger.info(f"TensorRT 輸出: {self.output_name}, 形狀: {self.output_shape}")
        logger.info(f"偵測到 {self.num_classes} 個類別 (YOLO {self.yolo_version})")
    
    def _allocate_buffers(self):
        """分配 GPU 和主機記憶體。"""
        # 計算記憶體大小
        input_size = int(np.prod(self.input_shape) * np.dtype(np.float32).itemsize)
        output_size = int(np.prod(self.output_shape) * np.dtype(np.float32).itemsize)
        
        # 分配 GPU 記憶體
        err, self.d_input = cudart.cudaMalloc(input_size)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"分配輸入 GPU 記憶體失敗: {err}")
        
        err, self.d_output = cudart.cudaMalloc(output_size)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"分配輸出 GPU 記憶體失敗: {err}")
        
        # 使用 pinned memory 加速 Host<->Device 傳輸
        # Pinned memory 不會被 OS 交換到磁碟，傳輸速度更快
        if self.use_pinned_memory:
            try:
                # 分配固定記憶體 (pinned memory)
                err, h_input_ptr = cudart.cudaMallocHost(input_size)
                if err == cudart.cudaError_t.cudaSuccess:
                    # 創建 numpy 數組視圖到 pinned memory (先創建1D再reshape)
                    h_input_flat = np.ctypeslib.as_array(
                        (np.ctypeslib.ctypes.c_float * int(np.prod(self.input_shape))).from_address(h_input_ptr)
                    )
                    self.h_input = h_input_flat.reshape(self.input_shape)
                    self._h_input_ptr = h_input_ptr
                    logger.info(f"已分配 pinned memory 用於輸入, 形狀: {self.h_input.shape}")
                else:
                    self.h_input = None
                    self._h_input_ptr = None
                    logger.warning(f"分配 pinned input memory 失敗: {err}，使用普通記憶體")
                
                err, h_output_ptr = cudart.cudaMallocHost(output_size)
                if err == cudart.cudaError_t.cudaSuccess:
                    h_output_flat = np.ctypeslib.as_array(
                        (np.ctypeslib.ctypes.c_float * int(np.prod(self.output_shape))).from_address(h_output_ptr)
                    )
                    self.h_output = h_output_flat.reshape(self.output_shape)
                    self._h_output_ptr = h_output_ptr
                    logger.info(f"已分配 pinned memory 用於輸出, 形狀: {self.h_output.shape}")
                else:
                    self.h_output = np.empty(self.output_shape, dtype=np.float32)
                    self._h_output_ptr = None
                    logger.warning(f"分配 pinned output memory 失敗: {err}，使用普通記憶體")
            except Exception as e:
                logger.warning(f"Pinned memory 分配出錯: {e}，回退到普通記憶體")
                self.h_input = None
                self._h_input_ptr = None
                self.h_output = np.empty(self.output_shape, dtype=np.float32)
                self._h_output_ptr = None
        else:
            self.h_input = None
            self._h_input_ptr = None
            self.h_output = np.empty(self.output_shape, dtype=np.float32)
            self._h_output_ptr = None
        
        # 設定 TensorRT 張量地址
        self.context.set_tensor_address(self.input_name, self.d_input)
        self.context.set_tensor_address(self.output_name, self.d_output)
        
        pinned_status = "已啟用" if (self._h_input_ptr or self._h_output_ptr) else "未啟用"
        logger.info(f"GPU 記憶體已分配: 輸入 {input_size/1024:.1f}KB, 輸出 {output_size/1024:.1f}KB (Pinned: {pinned_status})")
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """預處理影像。"""
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        hwc_shape = (input_h, input_w, 3)
        nchw_shape = (1, 3, input_h, input_w)

        if (
            self._preprocess_resize_buffer is None
            or self._preprocess_resize_buffer.shape != hwc_shape
        ):
            self._preprocess_resize_buffer = np.empty(hwc_shape, dtype=np.uint8)
            self._preprocess_float_buffer = np.empty(hwc_shape, dtype=np.float32)
            self._preprocess_input_buffer = np.empty(nchw_shape, dtype=np.float32)

        cv2.resize(
            frame,
            (input_w, input_h),
            dst=self._preprocess_resize_buffer,
            interpolation=cv2.INTER_LINEAR,
        )
        np.multiply(
            self._preprocess_resize_buffer,
            self._input_scale,
            out=self._preprocess_float_buffer,
            casting="unsafe",
        )
        self._preprocess_input_buffer[0] = np.transpose(
            self._preprocess_float_buffer, (2, 0, 1)
        )
        return self._preprocess_input_buffer
    
    def _infer(self, input_data: np.ndarray) -> np.ndarray:
        """執行 TensorRT 推理 (優化版)。"""
        # 如果有 pinned memory，使用它進行傳輸 (更快)
        if self.h_input is not None:
            # 直接複製到 pinned memory 然後傳輸
            self.h_input[:] = input_data
            cudart.cudaMemcpyAsync(
                self.d_input,
                self._h_input_ptr,
                input_data.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream
            )
        else:
            # 使用普通記憶體傳輸
            cudart.cudaMemcpyAsync(
                self.d_input,
                input_data.ctypes.data,
                input_data.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream
            )
        
        # 執行推理
        self.context.execute_async_v3(self.stream)
        
        # 複製輸出回主機 (使用 pinned memory 如果可用)
        if self._h_output_ptr is not None:
            cudart.cudaMemcpyAsync(
                self._h_output_ptr,
                self.d_output,
                self.h_output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream
            )
        else:
            cudart.cudaMemcpyAsync(
                self.h_output.ctypes.data,
                self.d_output,
                self.h_output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream
            )
        
        # 同步等待完成
        cudart.cudaStreamSynchronize(self.stream)
        
        # 使用 pinned memory 時不需要額外複製
        if self._h_output_ptr is not None:
            return self.h_output  # 直接返回，因為已經是有效數據
        return self.h_output.copy()
    
    def run_inference(self, frame: np.ndarray) -> tuple:
        """對單一幀執行 TensorRT 推理。"""
        try:
            start_time = time.perf_counter()
            
            # 應用色彩校正
            if self.image_processor:
                frame = self.image_processor.process(frame)
            
            # 預處理
            input_img = self._preprocess(frame)
            
            # 推理
            output = self._infer(input_img)
            
            inference_time = (time.perf_counter() - start_time) * 1000
            
            # 後處理 (根據 YOLO 版本)
            boxes, scores, class_ids = self._process_output(output)
            
            return (boxes, scores, class_ids), inference_time
            
        except Exception as e:
            logger.error(f"TensorRT 推論錯誤: {e}")
            logger.error(traceback.format_exc())
            return (np.array([]), np.array([]), np.array([])), 0.0
    
    def _process_output(self, output: np.ndarray) -> tuple:
        """根據 YOLO 版本處理輸出。"""
        if self.yolo_version == 'v5':
            # YOLOv5 格式: [batch, anchors, 5 + num_classes]
            predictions = output[0]
            boxes = predictions[:, :4]
            objectness = predictions[:, 4]
            class_probs = predictions[:, 5:5+self.num_classes]
            
            if np.min(objectness) < 0 or np.max(objectness) > 1:
                objectness = 1 / (1 + np.exp(-objectness))
            if np.min(class_probs) < 0 or np.max(class_probs) > 1:
                class_probs = 1 / (1 + np.exp(-class_probs))
            
            scores = objectness[:, np.newaxis] * class_probs
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
        else:
            # YOLOv8/v11/v12 格式: [batch, 4 + num_classes, anchors]
            predictions = output[0].T
            boxes = predictions[:, :4]
            class_probs = predictions[:, 4:4+self.num_classes]
            
            max_scores = np.max(class_probs, axis=1)
            class_ids = np.argmax(class_probs, axis=1)
        
        return boxes, max_scores, class_ids
    
    def cleanup(self) -> None:
        """清理 TensorRT 資源。"""
        try:
            if self.stream:
                cudart.cudaStreamDestroy(self.stream)
                self.stream = None
            
            if self.d_input:
                cudart.cudaFree(self.d_input)
                self.d_input = None
            
            if self.d_output:
                cudart.cudaFree(self.d_output)
                self.d_output = None
            
            # 釋放 pinned memory
            if hasattr(self, '_h_input_ptr') and self._h_input_ptr:
                cudart.cudaFreeHost(self._h_input_ptr)
                self._h_input_ptr = None
                self.h_input = None
            
            if hasattr(self, '_h_output_ptr') and self._h_output_ptr:
                cudart.cudaFreeHost(self._h_output_ptr)
                self._h_output_ptr = None
            
            self.context = None
            self.engine = None
            self.h_output = None
            self._preprocess_resize_buffer = None
            self._preprocess_float_buffer = None
            self._preprocess_input_buffer = None
            
            logger.info("TensorRT 資源已釋放")
        except Exception as e:
            logger.error(f"清理 TensorRT 資源時發生錯誤: {e}")


def is_tensorrt_available() -> bool:
    """檢查 TensorRT 是否可用。"""
    return TRT_AVAILABLE
