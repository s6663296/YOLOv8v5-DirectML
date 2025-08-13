import sys
import os

def resource_path(relative_path):
    """ 獲取資源的絕對路徑，適用於開發環境和 PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 創建一個暫存資料夾並將路徑儲存在 _MEIPASS 中
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)