from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen
import win32gui
import win32con

class AimOverlayWindow(QWidget):
    def __init__(self, aim_range=150, color=(0, 255, 255, 200), lock_color=(255, 0, 0, 200)):
        super().__init__()
        # 設定視窗屬性為透明背景
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # 設定視窗標誌：置頂、無邊框、允許滑鼠穿透、不在工作列顯示圖示
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.aim_range = aim_range
        self.target_in_range = False  # 新增：追蹤瞄準範圍內是否有目標
        self.set_colors(color, lock_color)
        # 根據瞄準範圍設定視窗大小，留一些邊距
        self.setFixedSize(int(self.aim_range * 2), int(self.aim_range * 2)) # 將視窗大小設定為瞄準範圍的兩倍（直徑），並轉換為整數
        self.updateGeometry() # 確保視窗幾何資訊更新
        # 移除初始化時的 center_on_screen() 呼叫，改為在 show() 後呼叫
        
    def showEvent(self, event):
        """在視窗顯示時應用穿透樣式"""
        super().showEvent(event)
        hwnd = self.winId()
        if hwnd:
            try:
                ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED)
            except Exception as e:
                print(f"Error setting window style for AimOverlay: {e}")
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

    def set_aim_range(self, new_range):
        """更新瞄準範圍並重繪視窗"""
        self.aim_range = new_range
        self.setFixedSize(int(self.aim_range * 2), int(self.aim_range * 2)) # 更新視窗大小為瞄準範圍的兩倍（直徑），並轉換為整數
        self.updateGeometry() # 確保視窗幾何資訊更新
        self.center_on_screen() # 重新置中
        if self.isVisible():
            self.update() # 觸發 paintEvent 重繪

    def set_colors(self, color, lock_color):
        """設定正常和鎖定時的顏色"""
        self.color = QColor(*color)
        self.lock_color = QColor(*lock_color)

    def paintEvent(self, event):
        """繪製瞄準範圍圓圈"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 獲取螢幕的設備像素比
        device_pixel_ratio = self.screen().devicePixelRatio() if self.screen() else 1.0

        # 根據是否有目標來設定圓圈顏色
        if self.target_in_range:
            circle_color = self.lock_color
        else:
            circle_color = self.color
        pen = QPen(circle_color)
        pen.setWidth(1)
        painter.setPen(pen)

        # 圓心在視窗的中心
        center_x = self.width() // 2
        center_y = self.height() // 2

        # 半徑是視窗寬度的一半，因為視窗大小已設為直徑 (aim_range * 2)
        # 我們繪製一個填滿整個元件的圓圈
        radius = self.width() // 2

        # 繪製橢圓以填滿元件, 減1是為了讓邊框線不會被裁切
        painter.drawEllipse(0, 0, self.width() - 1, self.height() - 1)

    def set_target_in_range(self, in_range: bool):
        """
        設定瞄準範圍內是否有目標的狀態，並觸發重繪。
        當 in_range 為 True 時，圓圈變為紅色；False 時恢復青藍色。
        """
        if self.target_in_range != in_range:
            self.target_in_range = in_range
            if self.isVisible():
                self.update() # 觸發 paintEvent 重繪
