# Module/thread_pool_manager.py
"""
用於優化滑鼠移動操作的執行緒池管理器。
以高效的執行緒池管理取代頻繁的執行緒創建。
"""
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any, Optional
from Module.logger import logger
import Module.control as control

class MouseMoveTask:
    """代表一個滑鼠移動任務。"""
    
    def __init__(self, mouse_move_mode: str, move_x: int, move_y: int, priority: int = 0):
        self.mouse_move_mode = mouse_move_mode
        self.move_x = move_x
        self.move_y = move_y
        self.priority = priority
        self.timestamp = time.time()
        
    def __lt__(self, other):
        """用於優先級佇列排序 - 優先級較高者優先，其次按時間戳。"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp
    
    def execute(self):
        """執行滑鼠移動。"""
        try:
            control.move(self.mouse_move_mode, self.move_x, self.move_y)
        except Exception as e:
            logger.error(f"執行滑鼠移動時出錯: {e}")

class ThreadPoolManager:
    """
    管理不同類型操作的執行緒池。
    針對滑鼠移動任務進行優化，以減少執行緒創建的開銷。
    """
    
    def __init__(self, max_mouse_workers: int = 2, max_general_workers: int = 4):
        """
        初始化執行緒池管理器。
        
        Args:
            max_mouse_workers: 滑鼠操作的最大執行緒數
            max_general_workers: 一般操作的最大執行緒數
        """
        self.max_mouse_workers = max_mouse_workers
        self.max_general_workers = max_general_workers
        
        self._mouse_executor = None
        self._general_executor = None
        
        self._mouse_task_queue = queue.PriorityQueue()
        self._shutdown_event = threading.Event()
        
        self._mouse_tasks_submitted = 0
        self._mouse_tasks_completed = 0
        self._mouse_tasks_dropped = 0
        self._last_stats_time = time.time()
        
        self._lock = threading.Lock()
        
        self._initialize_pools()
        
    def _initialize_pools(self):
        """初始化執行緒池。"""
        try:
            self._mouse_executor = ThreadPoolExecutor(
                max_workers=self.max_mouse_workers,
                thread_name_prefix="MouseMove"
            )
            
            self._general_executor = ThreadPoolExecutor(
                max_workers=self.max_general_workers,
                thread_name_prefix="General"
            )
            
            logger.info(f"執行緒池已初始化: 滑鼠={self.max_mouse_workers}, 一般={self.max_general_workers}")
            
        except Exception as e:
            logger.error(f"初始化執行緒池失敗: {e}")
            raise
    
    def submit_mouse_move(self, mouse_move_mode: str, move_x: int, move_y: int, priority: int = 0) -> bool:
        """
        向執行緒池提交一個滑鼠移動任務。
        
        Args:
            mouse_move_mode: 滑鼠移動模式
            move_x: X 軸移動量
            move_y: Y 軸移動量
            priority: 任務優先級（越高越重要）
            
        Returns:
            如果任務成功提交則為 True，否則為 False
        """
        if self._shutdown_event.is_set():
            return False
            
        try:
            with self._lock:
                self._mouse_tasks_submitted += 1
                
                if self._mouse_task_queue.qsize() > 10:
                    try:
                        old_task = self._mouse_task_queue.get_nowait()
                        self._mouse_tasks_dropped += 1
                        logger.debug("為防止佇列溢出，已丟棄舊的滑鼠任務")
                    except queue.Empty:
                        pass
                
                task = MouseMoveTask(mouse_move_mode, move_x, move_y, priority)
                
                future = self._mouse_executor.submit(self._execute_mouse_task, task)
                
                return True
                
        except Exception as e:
            logger.error(f"提交滑鼠移動任務失敗: {e}")
            return False
    
    def _execute_mouse_task(self, task: MouseMoveTask):
        """執行一個滑鼠移動任務。"""
        try:
            task.execute()
            with self._lock:
                self._mouse_tasks_completed += 1
        except Exception as e:
            logger.error(f"滑鼠任務執行出錯: {e}")
    
    def submit_general_task(self, func: Callable, *args, **kwargs) -> Optional[Future]:
        """
        向執行緒池提交一個一般任務。
        
        Args:
            func: 要執行的函式
            *args: 函式參數
            **kwargs: 函式關鍵字參數
            
        Returns:
            Future 物件或如果提交失敗則為 None
        """
        if self._shutdown_event.is_set():
            return None
            
        try:
            future = self._general_executor.submit(func, *args, **kwargs)
            return future
        except Exception as e:
            logger.error(f"提交一般任務失敗: {e}")
            return None
    
    def get_mouse_pool_stats(self) -> dict:
        """獲取滑鼠移動池的統計數據。"""
        with self._lock:
            current_time = time.time()
            time_elapsed = current_time - self._last_stats_time
            
            stats = {
                "tasks_submitted": self._mouse_tasks_submitted,
                "tasks_completed": self._mouse_tasks_completed,
                "tasks_dropped": self._mouse_tasks_dropped,
                "tasks_pending": self._mouse_tasks_submitted - self._mouse_tasks_completed,
                "queue_size": self._mouse_task_queue.qsize(),
                "max_workers": self.max_mouse_workers,
                "time_elapsed": time_elapsed
            }
            
            if time_elapsed > 0:
                stats["submission_rate"] = self._mouse_tasks_submitted / time_elapsed
                stats["completion_rate"] = self._mouse_tasks_completed / time_elapsed
            else:
                stats["submission_rate"] = 0
                stats["completion_rate"] = 0
                
            return stats
    
    def get_general_pool_stats(self) -> dict:
        """獲取一般執行緒池的統計數據。"""
        if not self._general_executor:
            return {"status": "not_initialized"}
            
        return {
            "max_workers": self.max_general_workers,
            "active_threads": len(self._general_executor._threads) if hasattr(self._general_executor, '_threads') else 0
        }
    
    def reset_stats(self):
        """重設統計計數器。"""
        with self._lock:
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
        
        try:
            if self._mouse_executor:
                self._mouse_executor.shutdown(wait=wait, timeout=timeout)
                logger.info("滑鼠執行緒池關閉完成")
                
            if self._general_executor:
                self._general_executor.shutdown(wait=wait, timeout=timeout)
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
                
            if not self._mouse_executor or self._mouse_executor._shutdown:
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