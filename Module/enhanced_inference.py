# Module/enhanced_inference.py
"""
增強的推論管理器，使用策略模式支援 YOLOv5/v8/v11/v12 ONNX 模型與 TensorRT 推理。
"""
import os
import time
import numpy as np
import onnxruntime as ort
import cv2
import traceback
import ast
from abc import ABC, abstractmethod
from Module.logger import logger
from Module.config import Config
from Module.inference_backend import (
    DEFAULT_BACKEND,
    get_effective_onnx_backend,
    get_onnx_providers,
    normalize_backend,
    should_use_tensorrt,
)

# TensorRT 策略導入
try:
    from Module.tensorrt_inference import TensorRTStrategy, is_tensorrt_available
    TRT_AVAILABLE = is_tensorrt_available()
except ImportError:
    TRT_AVAILABLE = False
    TensorRTStrategy = None

class InferenceStrategy(ABC):
    """推論策略的抽象基礎類別。"""
    def __init__(self, app_instance, image_processor=None):
        self.app = app_instance
        self.image_processor = image_processor
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None
        self.model_names = {}
        self.num_classes = 80
        self._preprocess_resize_buffer = None
        self._preprocess_float_buffer = None
        self._preprocess_input_buffer = None
        self._input_scale = np.float32(1.0 / 255.0)

    @abstractmethod
    def initialize(self) -> bool:
        """載入模型並準備推論。"""
        pass

    @abstractmethod
    def run_inference(self, frame: np.ndarray) -> tuple:
        """對單一幀執行推論。"""
        pass
    
    def _common_initialize(self) -> bool:
        """通用模型載入和初始化邏輯。"""
        model_path = Config.get("model_file")
        if not model_path:
            logger.error("在設定中未配置模型檔案路徑。")
            return False

        if not os.path.isabs(model_path):
            from Module.utils import resource_path
            model_path = resource_path(model_path)

        try:
            if not os.path.exists(model_path):
                logger.error(f"找不到 ONNX 模型檔案: {model_path}")
                return False

            logger.info(f"正在從以下路徑載入模型: {model_path}")
            
            # 效能優化：設定 ONNX Session 選項
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4  # 單一運算內的平行執行緒數
            sess_options.inter_op_num_threads = 2  # 運算之間的平行執行緒數
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = True  # 啟用記憶體競技場減少分配開銷
            sess_options.enable_mem_pattern = True  # 啟用記憶體模式優化重複使用
            
            # 根據設定選擇執行提供者
            configured_backend = normalize_backend(Config.get("inference_backend", DEFAULT_BACKEND))
            inference_backend = get_effective_onnx_backend(configured_backend)
            providers = get_onnx_providers(configured_backend)
            
            if configured_backend != inference_backend:
                logger.info(
                    f"設定後端為 {configured_backend}，ONNX Runtime 將使用 {inference_backend} (providers: {providers})"
                )
            else:
                logger.info(f"嘗試使用推理後端: {inference_backend} (providers: {providers})")
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            
            provider = self.session.get_providers()[0]
            logger.info(f"ONNX Runtime 正在使用 provider: {provider}")

            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_name = self.session.get_outputs()[0].name
            self.output_shape = self.session.get_outputs()[0].shape

            # 處理動態形狀：如果形狀包含字串（例如 'batch', 'height', 'width'），使用 model_size
            if isinstance(self.input_shape[2], str) or isinstance(self.input_shape[3], str):
                # 使用 app 中配置的 model_size
                actual_size = self.app.model_size if hasattr(self.app, 'model_size') and self.app.model_size else 320
                self.input_shape = [1, 3, actual_size, actual_size]
                logger.info(f"偵測到動態輸入形狀，使用固定大小: {self.input_shape}")
            else:
                self.app.model_size = self.input_shape[2]

            meta = self.session.get_modelmeta()
            if 'names' in meta.custom_metadata_map:
                try:
                    self.model_names = ast.literal_eval(meta.custom_metadata_map['names'])
                    logger.info(f"模型類別名稱已載入: {self.model_names}")
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"無法從元資料解析模型名稱: {e}。將使用數字 ID。")
                    self.model_names = {}
            else:
                logger.warning("在模型元資料中找不到類別名稱。將使用數字 ID。")

            logger.info(f"模型輸入名稱: {self.input_name}, 形狀: {self.input_shape}")
            logger.info(f"模型輸出名稱: {self.output_name}, 形狀: {self.output_shape}")

            logger.info("正在預熱 ONNX 模型...")
            dummy_input = np.zeros(self.input_shape, dtype=np.float32)
            for _ in range(5):
                self.session.run([self.output_name], {self.input_name: dummy_input})
            
            return True

        except Exception as e:
            logger.error(f"初始化 ONNX Runtime 引擎失敗: {e}")
            logger.error(traceback.format_exc())
            return False

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """通用預處理步驟。"""
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

    def cleanup(self) -> None:
        """清理資源。"""
        logger.info(f"{self.__class__.__name__} 資源已釋放")
        self.session = None
        self._preprocess_resize_buffer = None
        self._preprocess_float_buffer = None
        self._preprocess_input_buffer = None


class BackendUnavailableStrategy(InferenceStrategy):
    """用於回傳明確錯誤訊息的佔位策略。"""

    def __init__(self, app_instance, reason: str):
        super().__init__(app_instance)
        self.reason = reason

    def initialize(self) -> bool:
        logger.error(self.reason)
        return False

    def run_inference(self, frame: np.ndarray) -> tuple:
        return (np.array([]), np.array([]), np.array([])), 0.0

class YOLOv5Strategy(InferenceStrategy):
    """YOLOv5 的原生推論策略。"""
    def initialize(self) -> bool:
        logger.info("正在使用原生 YOLOv5 策略進行初始化...")
        if not self._common_initialize():
            return False
        
        if len(self.output_shape) == 3:
            # YOLOv5 格式: [batch, anchors, 5 + num_classes]
            self.num_classes = max(1, self.output_shape[2] - 5)
        else:
            self.num_classes = 80
            logger.warning(f"非預期的輸出形狀 {self.output_shape}。預設為 {self.num_classes} 個類別。")
        
        logger.info(f"YOLOv5 類別數量: {self.num_classes}")
        logger.info("原生 YOLOv5 ONNX 引擎初始化成功")
        return True

    def run_inference(self, frame: np.ndarray) -> tuple:
        try:
            start_time = time.perf_counter()
            
            # 應用色彩校正
            if self.image_processor:
                frame = self.image_processor.process(frame)

            input_img = self._preprocess(frame)
            output = self.session.run([self.output_name], {self.input_name: input_img})[0]
            inference_time = (time.perf_counter() - start_time) * 1000

            boxes, scores, class_ids = self._process_output(output)
            return (boxes, scores, class_ids), inference_time
        except Exception as e:
            logger.error(f"YOLOv5 推論錯誤: {e}")
            logger.error(traceback.format_exc())
            return (np.array([]), np.array([]), np.array([])), 0.0

    def _process_output(self, output: np.ndarray) -> tuple:
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
        return boxes, max_scores, class_ids

class YOLOv8Strategy(InferenceStrategy):
    """YOLOv8 的原生推論策略。"""
    def initialize(self) -> bool:
        logger.info("正在使用原生 YOLOv8 策略進行初始化...")
        if not self._common_initialize():
            return False
            
        if len(self.output_shape) == 3:
            # YOLOv8 格式: [batch, 4 + num_classes, anchors]
            self.num_classes = max(1, self.output_shape[1] - 4)
        else:
            self.num_classes = 80
            logger.warning(f"非預期的輸出形狀 {self.output_shape}。預設為 {self.num_classes} 個類別。")

        logger.info(f"YOLOv8 類別數量: {self.num_classes}")
        logger.info("原生 YOLOv8 ONNX 引擎初始化成功")
        return True

    def run_inference(self, frame: np.ndarray) -> tuple:
        try:
            start_time = time.perf_counter()

            # 應用色彩校正
            if self.image_processor:
                frame = self.image_processor.process(frame)

            input_img = self._preprocess(frame)
            output = self.session.run([self.output_name], {self.input_name: input_img})[0]
            inference_time = (time.perf_counter() - start_time) * 1000

            boxes, scores, class_ids = self._process_output(output)
            return (boxes, scores, class_ids), inference_time
        except Exception as e:
            logger.error(f"YOLOv8 推論錯誤: {e}")
            logger.error(traceback.format_exc())
            return (np.array([]), np.array([]), np.array([])), 0.0

    def _process_output(self, output: np.ndarray) -> tuple:
        predictions = output[0].T
        boxes = predictions[:, :4]
        class_probs = predictions[:, 4:4+self.num_classes]
        
        max_scores = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1)
        return boxes, max_scores, class_ids

class YOLOv11Strategy(InferenceStrategy):
    """YOLOv11 的原生推論策略。"""
    def initialize(self) -> bool:
        logger.info("正在使用原生 YOLOv11 策略進行初始化...")
        if not self._common_initialize():
            return False
            
        if len(self.output_shape) == 3:
            # YOLOv11 格式: [batch, 4 + num_classes, anchors]
            self.num_classes = max(1, self.output_shape[1] - 4)
        else:
            self.num_classes = 80
            logger.warning(f"非預期的輸出形狀 {self.output_shape}。預設為 {self.num_classes} 個類別。")

        logger.info(f"YOLOv11 類別數量: {self.num_classes}")
        logger.info("原生 YOLOv11 ONNX 引擎初始化成功")
        return True

    def run_inference(self, frame: np.ndarray) -> tuple:
        try:
            start_time = time.perf_counter()

            # 應用色彩校正
            if self.image_processor:
                frame = self.image_processor.process(frame)

            input_img = self._preprocess(frame)
            output = self.session.run([self.output_name], {self.input_name: input_img})[0]
            inference_time = (time.perf_counter() - start_time) * 1000

            boxes, scores, class_ids = self._process_output(output)
            return (boxes, scores, class_ids), inference_time
        except Exception as e:
            logger.error(f"YOLOv11 推論錯誤: {e}")
            logger.error(traceback.format_exc())
            return (np.array([]), np.array([]), np.array([])), 0.0

    def _process_output(self, output: np.ndarray) -> tuple:
        predictions = output[0].T
        boxes = predictions[:, :4]
        class_probs = predictions[:, 4:4+self.num_classes]
        
        max_scores = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1)
        return boxes, max_scores, class_ids

class YOLOv12Strategy(InferenceStrategy):
    """YOLOv12 的原生推論策略。"""
    def initialize(self) -> bool:
        logger.info("正在使用原生 YOLOv12 策略進行初始化...")
        if not self._common_initialize():
            return False
            
        if len(self.output_shape) == 3:
            # YOLOv12 格式: [batch, 4 + num_classes, anchors]
            self.num_classes = max(1, self.output_shape[1] - 4)
        else:
            self.num_classes = 80
            logger.warning(f"非預期的輸出形狀 {self.output_shape}。預設為 {self.num_classes} 個類別。")

        logger.info(f"YOLOv12 類別數量: {self.num_classes}")
        logger.info("原生 YOLOv12 ONNX 引擎初始化成功")
        return True

    def run_inference(self, frame: np.ndarray) -> tuple:
        try:
            start_time = time.perf_counter()

            # 應用色彩校正
            if self.image_processor:
                frame = self.image_processor.process(frame)

            input_img = self._preprocess(frame)
            output = self.session.run([self.output_name], {self.input_name: input_img})[0]
            inference_time = (time.perf_counter() - start_time) * 1000

            boxes, scores, class_ids = self._process_output(output)
            return (boxes, scores, class_ids), inference_time
        except Exception as e:
            logger.error(f"YOLOv12 推論錯誤: {e}")
            logger.error(traceback.format_exc())
            return (np.array([]), np.array([]), np.array([])), 0.0

    def _process_output(self, output: np.ndarray) -> tuple:
        predictions = output[0].T
        boxes = predictions[:, :4]
        class_probs = predictions[:, 4:4+self.num_classes]
        
        max_scores = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1)
        return boxes, max_scores, class_ids

class EnhancedInferenceManager:
    """推論管理器的上下文，根據 YOLO 版本和推理後端選擇並委派給相應的策略。"""
    
    def __init__(self, app_instance, yolo_version: str, image_processor=None):
        """
        初始化管理器並根據指定的 yolo_version 和 inference_backend 選擇策略。
        """
        self.yolo_version = yolo_version
        self.image_processor = image_processor
        self.app_instance = app_instance
        self.use_tensorrt = False
        self.backend = normalize_backend(Config.get("inference_backend", DEFAULT_BACKEND))
        
        # 檢查是否使用 TensorRT 後端
        model_path = Config.get("model_file", "")
        model_is_engine = str(model_path).lower().endswith('.engine')
        
        if should_use_tensorrt(self.backend, model_path):
            if TRT_AVAILABLE and TensorRTStrategy is not None:
                logger.info("使用 TensorRT 推理後端")
                self._strategy = TensorRTStrategy(app_instance, image_processor)
                self.use_tensorrt = True
            else:
                if model_is_engine:
                    self._strategy = BackendUnavailableStrategy(
                        app_instance,
                        "目前選擇的是 .engine 模型，但 TensorRT 環境不可用。"
                        "請依照 TensorRT_安裝指南.md 安裝 tensorrt 與 cuda-python。"
                    )
                else:
                    logger.warning("TensorRT 不可用，將回退到 ONNX Runtime")
                    self._select_onnx_strategy(app_instance, yolo_version, image_processor)
        else:
            self._select_onnx_strategy(app_instance, yolo_version, image_processor)
        
        # 代理屬性以便外部訪問
        self.input_shape = None
        self.model_names = {}
    
    def _select_onnx_strategy(self, app_instance, yolo_version: str, image_processor):
        """根據 YOLO 版本選擇 ONNX 策略。"""
        if yolo_version == 'v5':
            self._strategy: InferenceStrategy = YOLOv5Strategy(app_instance, image_processor)
        elif yolo_version == 'v8':
            self._strategy: InferenceStrategy = YOLOv8Strategy(app_instance, image_processor)
        elif yolo_version == 'v11':
            self._strategy: InferenceStrategy = YOLOv11Strategy(app_instance, image_processor)
        elif yolo_version == 'v12':
            self._strategy: InferenceStrategy = YOLOv12Strategy(app_instance, image_processor)
        else:
            raise ValueError(f"不支援的 YOLO 版本: '{yolo_version}'。必須是 'v5'、'v8'、'v11' 或 'v12'。")

    def initialize(self) -> bool:
        """初始化所選的推論策略。"""
        backend_name = "TensorRT" if self.use_tensorrt else f"{self.yolo_version.upper()} ONNX"
        logger.info(f"🔒 EnhancedInferenceManager 正在使用 {backend_name} 策略進行初始化。")
        is_success = self._strategy.initialize()
        if is_success:
            # 從策略中獲取初始化後的屬性
            self.input_shape = self._strategy.input_shape
            self.model_names = getattr(self._strategy, 'model_names', {})
        return is_success

    def run_inference(self, frame: np.ndarray) -> tuple:
        """將推論任務委派給當前的策略物件。"""
        return self._strategy.run_inference(frame)

    def cleanup(self) -> None:
        """清理當前策略的資源。"""
        if self._strategy:
            self._strategy.cleanup()
        logger.info("EnhancedInferenceManager 資源已釋放")
