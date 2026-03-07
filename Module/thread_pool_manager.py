# Module/thread_pool_manager.py
"""
用於優化滑鼠移動操作的執行緒池管理器。
滑鼠移動使用「原子覆蓋」策略，確保永遠只執行最新的移動指令。
一般任務仍使用 ThreadPoolExecutor。
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from Module.logger import logger
import Module.control as control


class ThreadPoolManager:
    """
    管理不同類型操作的執行緒池。
    滑鼠移動使用專用執行緒 + 原子覆蓋策略（非 Queue），
    確保永遠只執行最新的移動指令，舊指令被覆蓋丟棄。
    """
    
    def __init__(self, max_mouse_workers: int = 1, max_general_workers: int = 4):
        """
        初始化執行緒池管理器。
        
        Args:
            max_mouse_workers: 未使用（保留介面相容性），滑鼠移動固定使用 1 個專用執行緒
            max_general_workers: 一般操作的最大執行緒數
        """
        self.max_general_workers = max_general_workers
        
        # --- 滑鼠移動：原子覆蓋策略 ---
        # 只保存最新的移動指令，專用執行緒不斷讀取並執行
        self._mouse_lock = threading.Lock()
        self._mouse_pending_mode: Optional[str] = None
        self._mouse_pending_x: int = 0
        self._mouse_pending_y: int = 0
        self._mouse_has_pending: bool = False  # 是否有待執行的移動
        self._mouse_new_move_event = threading.Event()  # 通知專用執行緒有新指令
        self._mouse_thread: Optional[threading.Thread] = None
        
        # 一般任務執行緒池
        self._general_executor = None
        self._shutdown_event = threading.Event()
        
        # 統計
        self._mouse_tasks_submitted = 0
        self._mouse_tasks_completed = 0
        self._mouse_tasks_dropped = 0
        self._last_stats_time = time.time()
        self._stats_lock = threading.Lock()
        
        self._initialize_pools()
        
    def _initialize_pools(self):
        """初始化執行緒池和滑鼠專用執行緒。"""
        try:
            self._general_executor = ThreadPoolExecutor(
                max_workers=self.max_general_workers,
                thread_name_prefix="General"
            )
            
            # 啟動滑鼠移動專用執行緒
            self._mouse_thread = threading.Thread(
                target=self._mouse_consumer_loop,
                daemon=True,
                name="MouseMoveConsumer"
            )
            self._mouse_thread.start()
            
            logger.info(f"執行緒池已初始化: 滑鼠=原子覆蓋(1執行緒), 一般={self.max_general_workers}")
            
        except Exception as e:
            logger.error(f"初始化執行緒池失敗: {e}")
            raise
    
    def _mouse_consumer_loop(self):
        """
        滑鼠移動專用消費者執行緒。
        使用 Event 等待新指令，每次只執行最新的移動。
        """
        while not self._shutdown_event.is_set():
            # 等待新的移動指令（帶超時以便檢測 shutdown）
            if not self._mouse_new_move_event.wait(timeout=0.1):
                continue
            self._mouse_new_move_event.clear()
            
            # 取出最新的移動指令（原子讀取 + 清除）
            with self._mouse_lock:
                if not self._mouse_has_pending:
                    continue
                mode = self._mouse_pending_mode
                mx = self._mouse_pending_x
                my = self._mouse_pending_y
                self._mouse_has_pending = False
            
            # 執行滑鼠移動
            try:
                control.move(mode, mx, my)
                with self._stats_lock:
                    self._mouse_tasks_completed += 1
            except Exception as e:
                logger.error(f"滑鼠任務執行出錯: {e}")
    
    def submit_mouse_move(self, mouse_move_mode: str, move_x: int, move_y: int, priority: int = 0) -> bool:
        """
        提交滑鼠移動指令（原子覆蓋策略）。
        
        新指令會直接覆蓋舊的未執行指令，確保永遠只執行最新的。
        
        Args:
            mouse_move_mode: 滑鼠移動模式
            move_x: X 軸移動量
            move_y: Y 軸移動量
            priority: 未使用（保留介面相容性）
            
        Returns:
            如果指令成功寫入則為 True
        """
        if self._shutdown_event.is_set():
            return False
            
        with self._mouse_lock:
            # 如果有舊指令還沒被消費，計數為丟棄
            if self._mouse_has_pending:
                with self._stats_lock:
                    self._mouse_tasks_dropped += 1
            
            # 原子覆蓋：直接寫入最新值
            self._mouse_pending_mode = mouse_move_mode
            self._mouse_pending_x = move_x
            self._mouse_pending_y = move_y
            self._mouse_has_pending = True
            
            with self._stats_lock:
                self._mouse_tasks_submitted += 1
        
        # 通知消費者執行緒（在鎖外，避免死鎖）
        self._mouse_new_move_event.set()
        return True
    
    def get_mouse_pool_stats(self) -> dict:
        """獲取滑鼠移動池的統計數據。"""
        with self._stats_lock:
            current_time = time.time()
            time_elapsed = current_time - self._last_stats_time
            
            stats = {
                "tasks_submitted": self._mouse_tasks_submitted,
                "tasks_completed": self._mouse_tasks_completed,
                "tasks_dropped": self._mouse_tasks_dropped,
                "tasks_pending": 1 if self._mouse_has_pending else 0,
                "strategy": "atomic_overwrite",
                "time_elapsed": time_elapsed
            }
            
            if time_elapsed > 0:
                stats["submission_rate"] = self._mouse_tasks_submitted / time_elapsed
                stats["completion_rate"] = self._mouse_tasks_completed / time_elapsed
            else:
                stats["submission_rate"] = 0
                stats["completion_rate"] = 0
                
            return stats
    
    def reset_stats(self):
        """重設統計計數器。"""
        with self._stats_lock:
            self._mouse_tasks_submitted = 0
            self._mouse_tasks_completed = 0
            self._mouse_tasks_dropped = 0
            self._last_stats_time = time.time()
    
    def shutdown(self, wait: bool = True, timeout: float = 5.0):
        """
        關閉執行緒池。
        
        Args:
            wait: 是否等待待處理任務完成
            timeout: 等待關閉的最長時間
        """
        logger.info("正在關閉執行緒池...")
        self._shutdown_event.set()
        self._mouse_new_move_event.set()  # 喚醒滑鼠執行緒使其退出
        
        try:
            # 等待滑鼠專用執行緒結束
            if self._mouse_thread and self._mouse_thread.is_alive():
                self._mouse_thread.join(timeout=timeout)
                logger.info("滑鼠執行緒關閉完成")
                
            if self._general_executor:
                self._general_executor.shutdown(wait=wait)
                logger.info("一般執行緒池關閉完成")
                
            stats = self.get_mouse_pool_stats()
            logger.info(f"最終滑鼠池統計: {stats}")
            
        except Exception as e:
            logger.error(f"執行緒池關閉期間出錯: {e}")
    
    def is_healthy(self) -> bool:
        """檢查執行緒池是否健康。"""
        try:
            if self._shutdown_event.is_set():
                return False
                
            if not self._mouse_thread or not self._mouse_thread.is_alive():
                return False
                
            if not self._general_executor or self._general_executor._shutdown:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"檢查執行緒池健康狀況時出錯: {e}")
            return False
    
    def restart_if_needed(self):
        """如果執行緒池不健康，則重新啟動。"""
        if not self.is_healthy():
            logger.warning("執行緒池不健康，正在重新啟動...")
            try:
                self.shutdown(wait=False, timeout=1.0)
                self._shutdown_event.clear()
                self._initialize_pools()
                logger.info("執行緒池重新啟動成功")
            except Exception as e:
                logger.error(f"重新啟動執行緒池失敗: {e}")


# 全域執行緒池管理器實例
_global_thread_pool_manager = None
_manager_lock = threading.Lock()

def get_thread_pool_manager() -> ThreadPoolManager:
    """獲取全域執行緒池管理器實例。"""
    global _global_thread_pool_manager
    
    with _manager_lock:
        if _global_thread_pool_manager is None:
            _global_thread_pool_manager = ThreadPoolManager()
        return _global_thread_pool_manager

def shutdown_global_thread_pool():
    """關閉全域執行緒池管理器。"""
    global _global_thread_pool_manager
    
    with _manager_lock:
        if _global_thread_pool_manager is not None:
            _global_thread_pool_manager.shutdown()
            _global_thread_pool_manager = None
