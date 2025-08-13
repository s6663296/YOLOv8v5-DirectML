# Module/frame_buffer_pool.py
"""
用於記憶體優化的畫面緩衝區池。
透過重複使用畫面緩衝區來減少垃圾回收（GC）的壓力，而非頻繁複製。
"""
import threading
import numpy as np
from collections import deque
from Module.logger import logger

class FrameBufferPool:
    """
    一個可重複使用的畫面緩衝區池，以最小化記憶體分配和垃圾回收壓力。
    """
    
    def __init__(self, buffer_size: tuple, max_buffers: int = 10):
        """
        初始化畫面緩衝區池。
        
        Args:
            buffer_size: 畫面緩衝區的（高度、寬度、通道數）
            max_buffers: 池中保留的最大緩衝區數量
        """
        self.buffer_size = buffer_size
        self.max_buffers = max_buffers
        self.available_buffers = deque()
        self.in_use_buffers = set()
        self.lock = threading.Lock()
        
        # 預先分配一些緩衝區
        self._preallocate_buffers(min(3, max_buffers))
        
    def _preallocate_buffers(self, count: int):
        """預先分配一定數量的緩衝區。"""
        for _ in range(count):
            buffer = np.zeros(self.buffer_size, dtype=np.uint8)
            self.available_buffers.append(buffer)
        logger.debug(f"已預先分配 {count} 個大小為 {self.buffer_size} 的畫面緩衝區")
    
    def get_buffer(self) -> np.ndarray:
        """
        從池中獲取一個緩衝區。如果沒有可用的緩衝區，則創建一個新的。
        
        Returns:
            一個可供使用的 numpy 陣列緩衝區
        """
        with self.lock:
            if self.available_buffers:
                buffer = self.available_buffers.popleft()
            else:
                # 如果池為空，則創建新緩衝區
                buffer = np.zeros(self.buffer_size, dtype=np.uint8)
                logger.debug("已創建新的畫面緩衝區（池為空）")
            
            self.in_use_buffers.add(id(buffer))
            return buffer
    
    def return_buffer(self, buffer: np.ndarray):
        """
        將緩衝區返回池中以供重複使用。
        
        Args:
            buffer: 要返回池中的緩衝區
        """
        with self.lock:
            buffer_id = id(buffer)
            if buffer_id in self.in_use_buffers:
                self.in_use_buffers.remove(buffer_id)
                
                # 僅在未超過最大緩衝區數量時保留緩衝區
                if len(self.available_buffers) < self.max_buffers:
                    # 清空緩衝區以供重複使用
                    buffer.fill(0)
                    self.available_buffers.append(buffer)
                # 如果緩衝區過多，則讓其被垃圾回收
    
    def copy_to_buffer(self, source: np.ndarray, target_buffer: np.ndarray = None) -> np.ndarray:
        """
        將來源畫面複製到池中的緩衝區。
        
        Args:
            source: 要複製的來源畫面
            target_buffer: 可選的預分配目標緩衝區
            
        Returns:
            包含複製後畫面的緩衝區
        """
        if target_buffer is None:
            target_buffer = self.get_buffer()
        
        # 確保目標緩衝區具有正確的形狀
        if target_buffer.shape != source.shape:
            # 如果形狀不匹配，我們需要調整目標大小或獲取新緩衝區
            self.return_buffer(target_buffer)
            # 更新緩衝區大小並獲取新緩衝區
            self.buffer_size = source.shape
            target_buffer = self.get_buffer()
            target_buffer = np.zeros(source.shape, dtype=np.uint8)
        
        # 複製資料
        np.copyto(target_buffer, source)
        return target_buffer
    
    def get_pool_stats(self) -> dict:
        """獲取緩衝區池的統計數據。"""
        with self.lock:
            return {
                "available_buffers": len(self.available_buffers),
                "in_use_buffers": len(self.in_use_buffers),
                "buffer_size": self.buffer_size,
                "max_buffers": self.max_buffers
            }
    
    def cleanup(self):
        """清理池中的所有緩衝區。"""
        with self.lock:
            self.available_buffers.clear()
            self.in_use_buffers.clear()
        logger.info("畫面緩衝區池已清理")


class FrameManager:
    """
    使用緩衝區池優化管理畫面操作。
    """
    
    def __init__(self, initial_frame_size: tuple = None):
        """
        初始化畫面管理器。
        
        Args:
            initial_frame_size: 畫面緩衝區的初始大小（高度、寬度、通道數）
        """
        self.buffer_pool = None
        self.current_frame_size = initial_frame_size
        self.lock = threading.Lock()
        
        if initial_frame_size:
            self.buffer_pool = FrameBufferPool(initial_frame_size)
    
    def _ensure_buffer_pool(self, frame_shape: tuple):
        """確保緩衝區池存在且與畫面形狀匹配。"""
        if self.buffer_pool is None or self.current_frame_size != frame_shape:
            if self.buffer_pool:
                self.buffer_pool.cleanup()
            self.current_frame_size = frame_shape
            self.buffer_pool = FrameBufferPool(frame_shape)
    
    def get_processed_frame_buffer(self, source_frame: np.ndarray) -> np.ndarray:
        """
        獲取一個用於處理的緩衝區，該緩衝區是來源畫面的副本。
        這取代了頻繁的 frame.copy() 操作。
        
        Args:
            source_frame: 要複製的來源畫面
            
        Returns:
            包含來源畫面副本的緩衝區
        """
        with self.lock:
            self._ensure_buffer_pool(source_frame.shape)
            return self.buffer_pool.copy_to_buffer(source_frame)
    
    def get_display_frame_buffer(self, source_frame: np.ndarray) -> np.ndarray:
        """
        獲取一個用於顯示的緩衝區，該緩衝區是來源畫面的副本。
        
        Args:
            source_frame: 要複製的來源畫面
            
        Returns:
            包含來源畫面副本的緩衝區
        """
        with self.lock:
            self._ensure_buffer_pool(source_frame.shape)
            return self.buffer_pool.copy_to_buffer(source_frame)
    
    def return_frame_buffer(self, buffer: np.ndarray):
        """
        將畫面緩衝區返回池中以供重複使用。
        
        Args:
            buffer: 要返回的緩衝區
        """
        if self.buffer_pool:
            self.buffer_pool.return_buffer(buffer)
    
    def get_stats(self) -> dict:
        """獲取畫面管理器的統計數據。"""
        if self.buffer_pool:
            return self.buffer_pool.get_pool_stats()
        return {"status": "no_buffer_pool"}
    
    def cleanup(self):
        """清理畫面管理器。"""
        if self.buffer_pool:
            self.buffer_pool.cleanup()