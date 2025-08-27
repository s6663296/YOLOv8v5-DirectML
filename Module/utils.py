import sys
import os

import ctypes

def get_screen_scaling_factor():
    """ 獲取主螢幕的 DPI 縮放比例 """
    try:
        # 設定程序 DPI 感知，這對於正確獲取 DPI 至關重要
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        # 獲取主螢幕的 DPI
        dpi = ctypes.windll.user32.GetDpiForSystem()
        # 標準 DPI 是 96
        scaling_factor = dpi / 96.0
        return scaling_factor
    except Exception as e:
        print(f"無法獲取螢幕縮放比例: {e}")
        return 1.0

def resource_path(relative_path):
    """ 獲取資源的絕對路徑，適用於開發環境和 PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 創建一個暫存資料夾並將路徑儲存在 _MEIPASS 中
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)