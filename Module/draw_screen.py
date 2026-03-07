from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QRect
import win32gui
import win32con
from Module.logger import logger


class DrawScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.boxes_to_draw = []
        self._box_padding = 15
        self._redraw_margin = 6
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        screen_geometry = QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geometry)

    def _boxes_to_region(self, boxes):
        if not boxes:
            return QRect()

        union_rect = QRect()
        for box in boxes:
            x1, y1, x2, y2 = box
            left = int(min(x1, x2) - self._box_padding - self._redraw_margin)
            top = int(min(y1, y2) - self._box_padding - self._redraw_margin)
            right = int(max(x1, x2) + self._box_padding + self._redraw_margin)
            bottom = int(max(y1, y2) + self._box_padding + self._redraw_margin)

            rect = QRect(left, top, max(1, right - left + 1), max(1, bottom - top + 1))
            union_rect = rect if union_rect.isNull() else union_rect.united(rect)

        return union_rect.intersected(self.rect())

    def update_boxes(self, boxes):
        new_boxes = boxes or []
        if new_boxes == self.boxes_to_draw:
            return

        old_region = self._boxes_to_region(self.boxes_to_draw)
        new_region = self._boxes_to_region(new_boxes)
        self.boxes_to_draw = new_boxes

        if old_region.isNull() and new_region.isNull():
            return

        if old_region.isNull():
            dirty_region = new_region
        elif new_region.isNull():
            dirty_region = old_region
        else:
            dirty_region = old_region.united(new_region)

        if not dirty_region.isNull():
            self.update(dirty_region)

    def paintEvent(self, event):
        if not self.boxes_to_draw:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(0, 255, 0, 200))
        pen.setWidth(1)
        painter.setPen(pen)

        padding = self._box_padding
        clip_rect = event.rect()

        for box in self.boxes_to_draw:
            x1, y1, x2, y2 = box

            x1 -= padding
            y1 -= padding
            x2 += padding
            y2 += padding

            rect = QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if not rect.intersects(clip_rect):
                continue

            corner_length = 10
            radius = 8
            pen_width = 2
            pen.setWidth(pen_width)
            painter.setPen(pen)

            painter.drawArc(rect.left(), rect.top(), radius * 2, radius * 2, 90 * 16, 90 * 16)
            painter.drawLine(rect.left() + radius, rect.top(), rect.left() + corner_length, rect.top())
            painter.drawLine(rect.left(), rect.top() + radius, rect.left(), rect.top() + corner_length)

            painter.drawArc(rect.right() - radius * 2, rect.top(), radius * 2, radius * 2, 0 * 16, 90 * 16)
            painter.drawLine(rect.right() - radius, rect.top(), rect.right() - corner_length, rect.top())
            painter.drawLine(rect.right(), rect.top() + radius, rect.right(), rect.top() + corner_length)

            painter.drawArc(rect.left(), rect.bottom() - radius * 2, radius * 2, radius * 2, 180 * 16, 90 * 16)
            painter.drawLine(rect.left() + radius, rect.bottom(), rect.left() + corner_length, rect.bottom())
            painter.drawLine(rect.left(), rect.bottom() - radius, rect.left(), rect.bottom() - corner_length)

            painter.drawArc(rect.right() - radius * 2, rect.bottom() - radius * 2, radius * 2, radius * 2, 270 * 16, 90 * 16)
            painter.drawLine(rect.right() - radius, rect.bottom(), rect.right() - corner_length, rect.bottom())
            painter.drawLine(rect.right(), rect.bottom() - radius, rect.right(), rect.bottom() - corner_length)

    def show_overlay(self):
        self.show()

    def hide_overlay(self):
        self.hide()

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
                logger.error(f"Error setting window style for DrawScreen: {e}")
