from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontMetrics
import win32gui
import win32con
from collections import deque
from Module.logger import logger


class LatencyOverlay(QLabel):
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
        font.setPixelSize(18)
        font.setBold(True)
        self.setFont(font)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        metrics = QFontMetrics(self.font())
        self.setFixedSize(metrics.horizontalAdvance("延遲: 999.9 ms") + 6, metrics.height() + 4)

        self.setStyleSheet("color: #00FF00;")
        self._last_color = "#00FF00"
        self.setText("延遲: -- ms")
        self.position_in_corner()

        self._latency_history = deque(maxlen=30)
        self._max_history = 30
        self._latency_sum = 0.0
        self._last_text = "延遲: -- ms"

    def position_in_corner(self):
        if self.screen():
            screen_geometry = self.screen().geometry()
            x = screen_geometry.width() - self.width() - 10
            y = 10
            self.move(x, y)

    def update_latency(self, latency_ms: float):
        if len(self._latency_history) == self._max_history:
            self._latency_sum -= self._latency_history[0]

        self._latency_history.append(latency_ms)
        self._latency_sum += latency_ms

        avg_latency = self._latency_sum / len(self._latency_history)

        if avg_latency < 5:
            color = "#00FF00"
        elif avg_latency < 10:
            color = "#7FFF00"
        elif avg_latency < 16:
            color = "#FFFF00"
        elif avg_latency < 25:
            color = "#FFA500"
        else:
            color = "#FF0000"

        if color != self._last_color:
            self.setStyleSheet(f"color: {color};")
            self._last_color = color

        latency_text = f"延遲: {avg_latency:.1f} ms"
        if latency_text != self._last_text:
            self.setText(latency_text)
            self._last_text = latency_text

    def clear_history(self):
        self._latency_history.clear()
        self._latency_sum = 0.0
        self._last_text = "延遲: -- ms"
        self.setText(self._last_text)
        if self._last_color != "#00FF00":
            self.setStyleSheet("color: #00FF00;")
            self._last_color = "#00FF00"

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
                logger.error(f"Error setting window style for LatencyOverlay: {e}")
        self.position_in_corner()
