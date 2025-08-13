# Module/capture.py
"""
管理螢幕截圖並提供畫面進行推論。
"""
import threading
import time
from collections import deque
import mss
import numpy as np
import cv2
from Module.logger import logger

class ScreenCaptureManager:
    """管理螢幕截圖並提供畫面進行推論。"""
    
    def __init__(self, screen_bbox: dict, exit_event: threading.Event, new_frame_event: threading.Event, target_fps: int = 240):
        """初始化螢幕截圖管理器。"""
        self.screen_bbox = screen_bbox
        self.exit_event = exit_event
        self.new_frame_event = new_frame_event
        self.target_fps = target_fps
        
        self.frame_lock = threading.Lock()
        self.frame_queue = deque(maxlen=3)
        self.latest_frame = None
        self.capture_fps = 0
        self.restart_required = threading.Event()
        self.thread = None
        
    def start(self):
        """啟動螢幕截圖執行緒。"""
        self.thread = threading.Thread(target=self._capture_thread, daemon=True, name="capture_thread")
        self.thread.start()
        logger.info("螢幕截圖執行緒已啟動")
        
    def _capture_thread(self):
        """螢幕截圖執行緒函式。"""
        try:
            with mss.mss() as sct:
                frame_times = deque(maxlen=100)
                last_fps_display = time.time()
                
                while not self.exit_event.is_set() and not self.restart_required.is_set():
                    try:
                        start_time = time.time()
                        
                        frame_raw = sct.grab(self.screen_bbox)
                        # 原始資料為 RGB 格式，因此我們只需重新塑形。
                        frame_rgb = np.frombuffer(frame_raw.rgb, dtype=np.uint8).reshape((frame_raw.height, frame_raw.width, 3))
                        
                        with self.frame_lock:
                            self.latest_frame = frame_rgb
                            self.frame_queue.append(frame_rgb)
                            
                        self.new_frame_event.set()
                        
                        frame_time = time.time() - start_time
                        frame_times.append(frame_time)
                        
                        if time.time() - last_fps_display > 2.0:
                            if frame_times:
                                avg_frame_time = sum(frame_times) / len(frame_times)
                                self.capture_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                                logger.debug(f"螢幕截圖 FPS: {self.capture_fps:.1f}")
                            last_fps_display = time.time()
                            
                        target_frame_time = 1.0 / self.target_fps
                        sleep_time = max(0, target_frame_time - frame_time)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        
                    except Exception as e:
                        logger.error(f"螢幕截圖錯誤: {e}")
                        time.sleep(0.5)
                        
        except Exception as e:
            logger.error(f"致命的螢幕截圖錯誤: {e}")
            self.restart_required.set()
            
    def get_frame(self):
        """獲取最新的畫面。"""
        with self.frame_lock:
            if not self.frame_queue:
                return None
            # 返回引用以避免不必要的複製
            return self.frame_queue[-1]
            
    def stop(self):
        """發送訊號停止執行緒。"""
        self.exit_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("螢幕截圖已停止。")