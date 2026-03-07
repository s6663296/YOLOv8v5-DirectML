# Module/async_inference_pipeline.py
"""
非同步推論管道，使用專用推論執行緒實現擷取與推論的並行處理。
設計目標：打破串行瓶頸，提升 FPS 至 144+。
"""
import threading
import time
from typing import Optional, Tuple, Any
import numpy as np
from Module.logger import logger


class AsyncInferencePipeline:
    """
    非同步推論管道。
    
    使用專用執行緒持續處理最新幀，main loop 可以非阻塞地提交幀和獲取結果。
    採用 "latest-wins" 策略：如果新幀到達時舊幀還在處理，則跳過舊幀。
    """
    
    def __init__(self, inference_manager, exit_event: threading.Event):
        """
        初始化非同步推論管道。
        
        Args:
            inference_manager: EnhancedInferenceManager 實例
            exit_event: 應用程式退出事件
        """
        self.inference_manager = inference_manager
        self.exit_event = exit_event
        
        # 幀緩衝區（latest-wins）
        # 這裡直接交換 frame 參考，避免在 submit 階段做大型 np.copyto 造成鎖競爭
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_timestamp: Optional[float] = None
        self._frame_lock = threading.Lock()
        self._new_frame_event = threading.Event()
        
        # 結果緩衝區
        self._latest_results: Optional[Tuple[Any, float]] = None
        self._latest_results_timestamp: Optional[float] = None
        self._result_consumed = True  # 追蹤結果是否已被消費
        self._results_lock = threading.Lock()
        self._result_ready_event = threading.Event()  # 推論完成時通知主迴圈
        
        # 推論執行緒
        self._inference_thread: Optional[threading.Thread] = None
        self._running = False
        
        # 效能統計
        self._inference_count = 0
        self._skipped_frames = 0
        self._last_inference_time = 0.0
        
    def start(self) -> None:
        """啟動推論執行緒。"""
        if self._running:
            logger.warning("AsyncInferencePipeline: 推論執行緒已在運行中")
            return
            
        self._running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="AsyncInferenceThread"
        )
        self._inference_thread.start()
        logger.info("AsyncInferencePipeline: 推論執行緒已啟動")
        
    def stop(self) -> None:
        """停止推論執行緒。"""
        self._running = False
        self._new_frame_event.set()  # 喚醒等待中的執行緒
        
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=1.0)
            
        logger.info(f"AsyncInferencePipeline: 已停止 (推論: {self._inference_count}, 跳過: {self._skipped_frames})")
        
    def submit_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """
        提交幀進行推論（非阻塞）。
        
        如果前一幀還在處理中，將被新幀覆蓋（latest-wins 策略）。
        
        Args:
            frame: 要處理的幀
            timestamp: 幀擷取時間戳
        """
        with self._frame_lock:
            # 如果有舊幀還沒處理，計數跳過
            if self._pending_frame is not None:
                self._skipped_frames += 1
                
            # 直接使用最新 frame 參考（latest-wins）：
            # capture 端每幀都提供新陣列，這裡不再額外 copy，降低 CPU 與鎖持有時間
            self._pending_frame = frame
            self._pending_timestamp = timestamp
            
        # 通知推論執行緒有新幀
        self._new_frame_event.set()
        
    def get_results(self) -> Tuple[Optional[Tuple[Any, Any, Any]], float, Optional[float]]:
        """
        獲取最新的推論結果（非阻塞）。
        
        每個結果只會被返回一次，之後會返回 None 直到有新的推論完成。
        
        Returns:
            (detection_results, inference_time, timestamp) 或 (None, 0.0, None) 如果無新結果
        """
        with self._results_lock:
            # 如果結果已被消費或沒有結果，返回 None
            if self._result_consumed or self._latest_results is None:
                return None, 0.0, None
            # 標記結果已被消費
            self._result_consumed = True
            self._result_ready_event.clear()  # 清除事件以等待下一次結果
            return self._latest_results[0], self._latest_results[1], self._latest_results_timestamp
            
    def _inference_loop(self) -> None:
        """推論執行緒主迴圈。"""
        logger.info("AsyncInferencePipeline: 推論迴圈開始")
        
        while self._running and not self.exit_event.is_set():
            # 等待新幀（帶超時以便檢查 exit_event）
            if not self._new_frame_event.wait(timeout=0.1):
                continue
            self._new_frame_event.clear()
            
            # 取得待處理的幀
            with self._frame_lock:
                if self._pending_frame is None:
                    continue
                frame = self._pending_frame
                timestamp = self._pending_timestamp
                self._pending_frame = None
                self._pending_timestamp = None
            
            # 執行推論
            try:
                start_time = time.perf_counter()
                detection_results, inference_time = self.inference_manager.run_inference(frame)
                self._last_inference_time = (time.perf_counter() - start_time) * 1000
                
                # 儲存結果並標記為未消費
                with self._results_lock:
                    self._latest_results = (detection_results, inference_time)
                    self._latest_results_timestamp = timestamp
                    self._result_consumed = False  # 標記有新結果可用
                
                # 通知主迴圈有新結果可用（在鎖外設置以避免持鎖時間過長）
                self._result_ready_event.set()
                self._inference_count += 1
                
            except Exception as e:
                logger.error(f"AsyncInferencePipeline: 推論錯誤 - {e}")
                
        logger.info("AsyncInferencePipeline: 推論迴圈結束")
        
    def get_stats(self) -> dict:
        """獲取效能統計資料。"""
        return {
            "inference_count": self._inference_count,
            "skipped_frames": self._skipped_frames,
            "last_inference_time_ms": self._last_inference_time,
            "skip_rate": self._skipped_frames / max(1, self._inference_count + self._skipped_frames)
        }
