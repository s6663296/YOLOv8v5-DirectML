from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen
import win32gui
import win32con
from Module.logger import logger

class AimOverlayWindow(QWidget):
    def __init__(self, width=300, height=300, color=(0, 255, 255, 200), lock_color=(255, 0, 0, 200)):
        super().__init__()
        # 設定視窗屬性為透明背景
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # 設定視窗標誌：置頂、無邊框、允許滑鼠穿透、不在工作列顯示圖示
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.aim_width = width
        self.aim_height = height
        self.target_in_range = False  # 新增：追蹤瞄準範圍內是否有目標
        self.set_colors(color, lock_color)
        # 根據瞄準範圍設定視窗大小，留一些邊距
        self._update_window_size_for_dpi()
        self.updateGeometry() # 確保視窗幾何資訊更新
        # 移除初始化時的 center_on_screen() 呼叫，改為在 show() 後呼叫
        
    def _update_window_size_for_dpi(self):
        """根據 aim_width/height 和螢幕 DPI 計算並設定視窗大小。"""
        # 優先使用元件所在的螢幕，如果不可用，則使用主螢幕
        screen = self.screen() or QApplication.primaryScreen()
        device_pixel_ratio = screen.devicePixelRatio() if screen else 1.0
        
        # aim_width/height 被視為物理像素，我們需要計算對應的邏輯像素大小
        logical_width = self.aim_width / device_pixel_ratio
        logical_height = self.aim_height / device_pixel_ratio
        
        self.setFixedSize(int(logical_width), int(logical_height))

    def showEvent(self, event):
        """在視窗顯示時應用穿透樣式並設定排除截圖"""
        super().showEvent(event)
        hwnd = self.winId()
        if hwnd:
            try:
                # 設定滑鼠穿透和透明
                ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED)
                
            except Exception as e:
                logger.error(f"Error setting window style for AimOverlay: {e}")
        self.center_on_screen()

    def center_on_screen(self):
        """將視窗移動到螢幕中心，考慮高DPI"""
        screen = QApplication.primaryScreen()
        # 獲取螢幕的邏輯像素尺寸
        screen_width = screen.size().width()
        screen_height = screen.size().height()

        # 獲取視窗的邏輯像素尺寸
        window_width = self.width()
        window_height = self.height()

        # 計算視窗左上角座標，使其中心位於螢幕中心（邏輯像素）
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 2)
        self.move(x, y)

    def update_size(self, width, height):
        """更新瞄準範圍並重繪視窗"""
        self.aim_width = width
        self.aim_height = height
        self._update_window_size_for_dpi()
        self.updateGeometry() # 確保視窗幾何資訊更新
        self.center_on_screen() # 重新置中
        if self.isVisible():
            self.update() # 觸發 paintEvent 重繪

    def set_colors(self, color, lock_color):
        """設定正常和鎖定時的顏色"""
        self.color = QColor(*color)
        self.lock_color = QColor(*lock_color)

    def paintEvent(self, event):
        """繪製瞄準範圍矩形"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 根據是否有目標來設定顏色
        if self.target_in_range:
            draw_color = self.lock_color
        else:
            draw_color = self.color
        pen = QPen(draw_color)
        pen.setWidth(2) # 稍微加粗一點邊框
        painter.setPen(pen)

        # 繪製矩形以填滿元件, 減1是為了讓邊框線不會被裁切
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

    def set_target_in_range(self, in_range: bool):
        """
        設定瞄準範圍內是否有目標的狀態，並觸發重繪。
        """
        if self.target_in_range != in_range:
            self.target_in_range = in_range
            if self.isVisible():
                self.update() # 觸發 paintEvent 重繪
