# Module/draw_screen.py
"""
此模組包含 DrawScreen 類別，負責建立透明覆蓋層以顯示 YOLO 偵測框。
"""
import sys
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QRect, QPoint
import win32gui
import win32con

class DrawScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.boxes_to_draw = []
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool # 避免在工作列中顯示
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 獲取全螢幕的幾何尺寸
        screen_geometry = QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geometry)

    def update_boxes(self, boxes):
        """
        更新要繪製的方框列表。
        Args:
            boxes (list): 一個元組列表，其中每個元組為 (x1, y1, x2, y2)。
        """
        self.boxes_to_draw = boxes
        self.update() # 觸發重繪

    def paintEvent(self, event):
        """
        在小工具上繪製方框。
        """
        if not self.boxes_to_draw:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 設定方框的畫筆顏色（例如：綠色）
        pen = QPen(QColor(0, 255, 0, 200)) # 帶有透明度的綠色
        pen.setWidth(1)
        painter.setPen(pen)

        # 設定一個 padding 值來擴大偵測方塊，讓方塊不要緊貼目標
        padding = 15

        for box in self.boxes_to_draw:
            x1, y1, x2, y2 = box

            # 擴大方框
            x1 -= padding
            y1 -= padding
            x2 += padding
            y2 += padding

            rect = QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            
            # 改為繪製邊角
            corner_length = 10  # 邊角的長度
            radius = 8 # 圓角半徑
            pen_width = 2 # 線條寬度
            pen.setWidth(pen_width)
            painter.setPen(pen)

            # 左上角
            painter.drawArc(rect.left(), rect.top(), radius * 2, radius * 2, 90 * 16, 90 * 16)
            painter.drawLine(rect.left() + radius, rect.top(), rect.left() + corner_length, rect.top())
            painter.drawLine(rect.left(), rect.top() + radius, rect.left(), rect.top() + corner_length)
            
            # 右上角
            painter.drawArc(rect.right() - radius*2, rect.top(), radius * 2, radius * 2, 0 * 16, 90 * 16)
            painter.drawLine(rect.right() - radius, rect.top(), rect.right() - corner_length, rect.top())
            painter.drawLine(rect.right(), rect.top() + radius, rect.right(), rect.top() + corner_length)

            # 左下角
            painter.drawArc(rect.left(), rect.bottom() - radius*2, radius * 2, radius * 2, 180 * 16, 90 * 16)
            painter.drawLine(rect.left() + radius, rect.bottom(), rect.left() + corner_length, rect.bottom())
            painter.drawLine(rect.left(), rect.bottom() - radius, rect.left(), rect.bottom() - corner_length)
            
            # 右下角
            painter.drawArc(rect.right() - radius*2, rect.bottom() - radius*2, radius * 2, radius * 2, 270 * 16, 90 * 16)
            painter.drawLine(rect.right() - radius, rect.bottom(), rect.right() - corner_length, rect.bottom())
            painter.drawLine(rect.right(), rect.bottom() - radius, rect.right(), rect.bottom() - corner_length)

    def show_overlay(self):
        self.show()

    def hide_overlay(self):
        self.hide()

    def showEvent(self, event):
        """在視窗顯示時應用穿透樣式"""
        super().showEvent(event)
        hwnd = self.winId()
        if hwnd:
            try:
                ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED)
            except Exception as e:
                print(f"為 DrawScreen 設定視窗樣式時出錯: {e}")
