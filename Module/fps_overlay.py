from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontMetrics
import win32gui
import win32con
from Module.logger import logger


class FPSOverlay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        font = QFont("Arial")
        font.setPixelSize(20)
        font.setBold(True)
        self.setFont(font)
        self.setStyleSheet("color: red;")
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        metrics = QFontMetrics(self.font())
        self.setFixedSize(metrics.horizontalAdvance("FPS: 9999") + 6, metrics.height() + 4)

        self._last_fps = None
        self.setText("FPS: --")
        self.position_in_corner()

    def position_in_corner(self):
        if self.screen():
            screen_geometry = self.screen().geometry()
            self.move(10, screen_geometry.height() - self.height() - 40)

    def update_fps(self, fps: int):
        if fps == self._last_fps:
            return
        self._last_fps = fps
        self.setText(f"FPS: {fps}")

    def showEvent(self, event):
        super().showEvent(event)
        hwnd = self.winId()
        if hwnd:
            try:
                ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                win32gui.SetWindowLong(
                    hwnd,
                    win32con.GWL_EXSTYLE,
                    ex_style | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED,
                )
            except Exception as e:
                logger.error(f"Error setting window style for FPSOverlay: {e}")
        self.position_in_corner()
