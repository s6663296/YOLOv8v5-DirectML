# Module/image_processor.py
"""
此模組提供 ImageProcessor 類別，用於在模型推論前對影像進行色彩校正與轉換。
功能包含 BGR/RGB 轉換、CLAHE、Gamma 校正、白平衡等，
所有操作皆可透過設定檔進行配置。
"""
import cv2
import numpy as np
from Module.logger import logger
from Module.config import Config

class ImageProcessor:
    """管理影像前處理流程，包括色彩校正與轉換。"""

    def __init__(self):
        """初始化影像處理器並從設定檔載入設定。"""
        self.load_config()
        logger.info("影像處理器 (ImageProcessor) 初始化完成。")

    def load_config(self):
        """從設定檔載入影像處理相關設定。"""
        self.bgr_to_rgb = Config.get("preprocess_bgr_to_rgb", True)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        對輸入的影像幀執行完整的處理流程。
        輸入影像為 BGR 格式 (來自 mss 擷取的 raw BGRA 去除 alpha)。
        """
        # 如果模型需要 RGB 輸入，執行一次 BGR→RGB 轉換
        if self.bgr_to_rgb:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 如果模型需要 BGR 輸入，直接返回（已經是 BGR）
        return frame
