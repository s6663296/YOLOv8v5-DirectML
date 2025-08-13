from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
import win32gui
import win32con

class FPSOverlay(QLabel):
    """一個 QLabel 小工具，用於將 FPS 顯示為永遠在最上層的覆蓋層。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        self.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.setStyleSheet("color: red;")
        self.setText("FPS: --")
        self.adjustSize()
        self.position_in_corner()

    def position_in_corner(self):
        """將小工具定位在螢幕的左下角。"""
        if self.screen():
            screen_geometry = self.screen().geometry()
            self.move(10, screen_geometry.height() - self.height() - 40) # 離左邊 10px，離底部 40px

    def update_fps(self, fps: int):
        """用於更新顯示的 FPS 的插槽。"""
        self.setText(f"FPS: {fps}")
        self.adjustSize() # 如果文本長度改變，則調整大小

    def showEvent(self, event):
        """確保顯示時位置正確並應用透明樣式。"""
        super().showEvent(event)
        hwnd = self.winId()
        if hwnd:
            try:
                ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED)
            except Exception as e:
                print(f"為 FPSOverlay 設定視窗樣式時出錯: {e}")
        self.position_in_corner()