import sys
import os
import ctypes
import cv2
import numpy as np
import pyautogui
import mss
import time
import threading
import psutil
from math import sqrt

# ---- 必須在 PyQt6 之前匯入 onnxruntime ----
# PyQt6 會修改 Windows DLL 搜尋路徑，導致 onnxruntime-directml 的
# DirectML.dll 載入失敗 (DLL 初始化例行程序失敗)。
# 預先載入 onnxruntime 可避免此問題。
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass

from PyQt6 import uic
import queue
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon, QCursor
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QScrollArea
import atexit

from Module.logger import logger
from Module.settings_manager import SettingsManager
from Module.tray_manager import TrayManager
from Module.fps_overlay import FPSOverlay
from Module.latency_overlay import LatencyOverlay
from Module.draw_screen import DrawScreen
from Module.capture import ScreenCaptureManager
from Module.enhanced_inference import EnhancedInferenceManager
from Module.async_inference_pipeline import AsyncInferencePipeline
from Module.recoil_control import RecoilControl
from Module.frame_buffer_pool import FrameManager
from Module.thread_pool_manager import get_thread_pool_manager, shutdown_global_thread_pool
from Module.new_aim_logic import NewAimLogic
from Module.ui_handler import UIHandler
from Module.preview_window import PreviewWindow
from Module.utils import resource_path, get_screen_scaling_factor

from Module.keyboard import keyboard_listener, get_key_code_vk
from Module.image_processor import ImageProcessor

class _NullStatusBar:
    """No-op status bar proxy to disable bottom status messages."""
    def showMessage(self, *_args, **_kwargs):
        return None

    def clearMessage(self):
        return None

class main(QMainWindow):
    aim_dimensions_changed = pyqtSignal(int, int)
    visual_offset_changed = pyqtSignal(float)
    fps_updated = pyqtSignal(int)
    latency_updated = pyqtSignal(float)
    aimbot_hotkey_toggle_requested = pyqtSignal()
 
    def __init__(self):
        super().__init__()
        self._null_status_bar = _NullStatusBar()
        uic.loadUi(resource_path("ui/main_window.ui"), self)
        self._load_tab_widgets()
        self.setWindowIcon(QIcon(resource_path("app.ico")))
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.centralWidget().setMouseTracking(True)

        # --- 響應式視窗 (RWD) 設定 ---
        self.RESIZE_MARGIN = 8  # 邊緣拖拉偵測範圍 (像素)
        self._resize_direction = None  # 當前拖拉調整方向
        self._resize_origin = None  # 拖拉起始點
        self._resize_geometry = None  # 拖拉起始時的視窗幾何
        self.setMinimumSize(350, 500)

        # 初始化 UI 處理器

        self.ui_handler = UIHandler(self)
        self.gridLayout_recoil.setColumnStretch(0, 0)
        self.gridLayout_recoil.setColumnStretch(1, 1)
        self.gridLayout_recoil.setColumnStretch(2, 0)
        
        self.real_screen_width, self.real_screen_height = pyautogui.size()
        self.available_monitors = []
        

        
        self.settings_manager = SettingsManager(self)
        
        # 初始化屬性
        self.model = None
        self.yolo_enabled = False
        self.preview_enabled = False
        self.aim_overlay_enabled = False
        self.aim_overlay_window = None
        self.tracked_target = None
        self.last_target_time = 0
        self.current_config_path = None

        self.is_zoomed = False # 追蹤自動縮放狀態
        
        # PID 控制器設定
        self.pid_enabled = False
        self.pid_kp = 50   # 0-100 整數，UI 中轉換為 0.0-1.0
        self.pid_ki = 0    # 0-50 整數，UI 中轉換為 0.0-0.05
        self.pid_kd = 10   # 0-100 整數，UI 中轉換為 0.0-0.1
        
        # 從 SettingsManager 載入初始設定
        self.settings_manager.load_initial_settings()

        self.recoil_control = RecoilControl(self)
        self.ui_handler.init_signals()
        
        self.image_processor = ImageProcessor()
        
        self.close_button.clicked.connect(self.close)
        self.minimize_button.clicked.connect(self.showMinimized)
        self.mode_switch_button.hide()

        self.fps_overlay_window = FPSOverlay()
        self.fps_updated.connect(self.fps_overlay_window.update_fps)

        self.latency_overlay_window = LatencyOverlay()
        self.latency_updated.connect(self.latency_overlay_window.update_latency)

        self.draw_screen_window = DrawScreen()

        self.yolo_version_combobox.currentTextChanged.connect(self.on_yolo_version_changed)

        # 內建模型相關UI元件已在 .ui 檔案中移除
        
        # 連接新的尺寸變更信號
        self.aim_dimensions_changed.connect(self.update_aim_dimensions_ui)

        self.export_button.clicked.connect(self.ui_handler.export_settings)
        self.import_button.clicked.connect(self.ui_handler.import_settings)
        self.save_config_button.clicked.connect(self.ui_handler.save_current_config)

        self.preview_window = PreviewWindow()
        self.preview_window.hide()
        self.toggle_preview_button.clicked.connect(self.toggle_preview_window)
        self.aimbot_hotkey_toggle_requested.connect(self._on_aimbot_hotkey_toggle_requested)

        self._aimbot_hotkey_thread = None
        self._aimbot_hotkey_stop_event = threading.Event()
        self._aimbot_hotkey_vk_code = None
        self.boxes_to_draw_queue = queue.Queue(maxsize=1)
        
        self.load_settings()
        self.update_aimbot_hotkey_listener()
        
        self.settings_manager.update_config_display()
        
        self.resize(470, 800)
        self.setMouseTracking(True)  # 確保 resize 後仍追蹤滑鼠
        
        self.exit_event = threading.Event()
        self.new_frame_event = threading.Event()
        self.boxes_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        
        self.capture_manager = None
        self.inference_manager = None
        self.async_pipeline = None  # 非同步推論管道
        self.processing_thread = None
        self.latest_processed_frame = None
        
        self.frame_manager = FrameManager()
        self.thread_pool_manager = get_thread_pool_manager()
        
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_display)

        self.drawing_timer = QTimer(self)
        self.drawing_timer.timeout.connect(self.update_drawing_overlay)
        
        self.thread_pool_health_timer = QTimer(self)
        self.thread_pool_health_timer.timeout.connect(self.check_thread_pool_health)
        self.thread_pool_health_timer.start(30000)
        
        self.aim_scale_timer = QTimer(self)
        self.aim_scale_timer.timeout.connect(self.check_aim_scale)
        
        self.capture_area = self.calculate_capture_area()
 
        self.new_aim_logic = NewAimLogic(self)
        self.nms_detections = None  # NMS 後的偵測結果 (xyxy, scores, class_ids)
 
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
        
        # 延遲覆蓋層更新頻率控制（每秒更新10次，避免過於頻繁）
        self._latency_update_interval = 0.1  # 100ms 更新一次
        self._last_latency_update_time = 0
 

        
        self.tray_manager = TrayManager(self)
        
        self.drag_position = None
        self._resize_direction = None
        self._last_cursor_direction = None
        self.screen_scaling_factor = get_screen_scaling_factor()
        logger.info(f"偵測到螢幕縮放比例: {self.screen_scaling_factor}")
        
        self.toggle_preview_button.setText("關閉 YOLO 預覽" if self.preview_enabled else "開啟 YOLO 預覽")
        


        self.set_process_priority()

    def statusBar(self):
        return self._null_status_bar

    def get_ui_refresh_interval(self, capture_fps: int) -> int:
        """限制 UI 刷新率，避免高 FPS 設定下 UI 執行緒過載。"""
        ui_fps_cap = 120
        effective_fps = max(1, min(int(capture_fps), ui_fps_cap))
        return max(1, 1000 // effective_fps)

    def _get_aimbot_hotkey_vk_code(self):
        hotkey_text = getattr(self, 'aimbot_toggle_hotkey', '')
        if not hotkey_text:
            return None

        hotkey_text = hotkey_text.strip().upper()
        if not hotkey_text or '+' in hotkey_text:
            return None

        vk_code = get_key_code_vk(hotkey_text)
        if vk_code == "UNKNOWN":
            return None
        return vk_code

    def _aimbot_hotkey_loop(self):
        last_pressed = False
        while not self._aimbot_hotkey_stop_event.is_set():
            vk_code = self._aimbot_hotkey_vk_code
            is_pressed = bool(vk_code is not None and keyboard_listener.is_pressed(vk_code))

            if is_pressed and not last_pressed:
                self.aimbot_hotkey_toggle_requested.emit()

            last_pressed = is_pressed
            self._aimbot_hotkey_stop_event.wait(0.01)

    def _play_aimbot_toggle_sound(self, enabled):
        beep_type = 0x00000040 if enabled else 0x00000010
        try:
            ctypes.windll.user32.MessageBeep(beep_type)
        except Exception as e:
            logger.debug(f"Failed to play aimbot toggle sound: {e}")

    def _on_aimbot_hotkey_toggle_requested(self):
        if not getattr(self, 'aimbot_hotkey_enabled', False):
            return
        if hasattr(self, 'aimbot_checkbox'):
            new_state = not self.aimbot_checkbox.isChecked()
            self.aimbot_checkbox.setChecked(new_state)
            self._play_aimbot_toggle_sound(new_state)

    def stop_aimbot_hotkey_listener(self):
        self._aimbot_hotkey_stop_event.set()

        if self._aimbot_hotkey_thread and self._aimbot_hotkey_thread.is_alive():
            self._aimbot_hotkey_thread.join(timeout=0.5)

        self._aimbot_hotkey_thread = None
        self._aimbot_hotkey_vk_code = None

    def update_aimbot_hotkey_listener(self):
        enabled = bool(getattr(self, 'aimbot_hotkey_enabled', False))
        vk_code = self._get_aimbot_hotkey_vk_code()

        if enabled and vk_code is None:
            logger.warning("Aimbot toggle hotkey is enabled but key is invalid.")
        if not enabled or vk_code is None:
            self.stop_aimbot_hotkey_listener()
            return

        is_running = self._aimbot_hotkey_thread and self._aimbot_hotkey_thread.is_alive()
        same_key = (self._aimbot_hotkey_vk_code == vk_code)
        if is_running and same_key:
            return

        self.stop_aimbot_hotkey_listener()
        self._aimbot_hotkey_vk_code = vk_code
        self._aimbot_hotkey_stop_event.clear()
        self._aimbot_hotkey_thread = threading.Thread(
            target=self._aimbot_hotkey_loop,
            name="AimbotHotkeyListener",
            daemon=True
        )
        self._aimbot_hotkey_thread.start()

    def _load_tab_widgets(self):
        """動態載入各個 Tab 分頁的 UI 並將子元件複製為 main window 的屬性"""
        from PyQt6.QtWidgets import QWidget
        from PyQt6.QtCore import QObject
        
        tab_files = [
            ("ui/aim_tab.ui", "瞄準"),
            ("ui/detection_tab.ui", "偵測"),
            ("ui/recoil_tab.ui", "壓槍"),
            ("ui/range_tab.ui", "範圍"),
            ("ui/general_tab.ui", "通用"),
        ]
        
        for ui_file, tab_title in tab_files:
            tab_widget = QWidget()
            uic.loadUi(resource_path(ui_file), tab_widget)

            # 用 QScrollArea 包裹 tab 內容，縮小視窗時可捲動
            scroll_area = QScrollArea()
            scroll_area.setObjectName("tab_scroll_area")
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(tab_widget)
            scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll_area.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            scroll_area.viewport().setAutoFillBackground(False)
            self.tabWidget.addTab(scroll_area, tab_title)
            
            # 將 tab 內的所有子物件（包括 QWidget 和 QLayout）複製為 main window 的屬性
            # 這樣現有的信號連接程式碼不需要修改
            for child in tab_widget.findChildren(QObject):
                if child.objectName():
                    setattr(self, child.objectName(), child)

    def refresh_capture_monitors(self):
        """偵測目前可用顯示器並更新下拉選單。"""
        monitors = []

        try:
            with mss.mss() as sct:
                for index, monitor in enumerate(sct.monitors[1:], start=1):
                    left = int(monitor.get("left", 0))
                    top = int(monitor.get("top", 0))
                    width = int(monitor.get("width", self.real_screen_width))
                    height = int(monitor.get("height", self.real_screen_height))
                    label = f"顯示器 {index} ({width}x{height}, {left},{top})"
                    monitors.append({
                        "index": index,
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                        "label": label,
                    })
        except Exception as e:
            logger.warning(f"無法列舉顯示器，將使用預設顯示器: {e}")

        if not monitors:
            monitors = [{
                "index": 1,
                "left": 0,
                "top": 0,
                "width": int(self.real_screen_width),
                "height": int(self.real_screen_height),
                "label": f"顯示器 1 ({int(self.real_screen_width)}x{int(self.real_screen_height)}, 0,0)",
            }]

        self.available_monitors = monitors

        try:
            selected_index = int(getattr(self, 'capture_monitor_index', 1))
        except (TypeError, ValueError):
            selected_index = 1

        selected_index = max(1, min(selected_index, len(self.available_monitors)))
        self.capture_monitor_index = selected_index

        if hasattr(self, 'capture_monitor_combobox'):
            self.capture_monitor_combobox.blockSignals(True)
            self.capture_monitor_combobox.clear()
            for monitor in self.available_monitors:
                self.capture_monitor_combobox.addItem(monitor['label'], monitor['index'])
            combo_idx = self.capture_monitor_combobox.findData(self.capture_monitor_index)
            self.capture_monitor_combobox.setCurrentIndex(max(0, combo_idx))
            self.capture_monitor_combobox.blockSignals(False)

        logger.info(f"已偵測到 {len(self.available_monitors)} 個顯示器")

    def get_monitor_by_index(self, monitor_index):
        """取得指定索引的顯示器資訊。"""
        if not hasattr(self, 'available_monitors') or not self.available_monitors:
            self.refresh_capture_monitors()

        try:
            normalized_index = int(monitor_index)
        except (TypeError, ValueError):
            normalized_index = 1

        normalized_index = max(1, min(normalized_index, len(self.available_monitors)))
        return self.available_monitors[normalized_index - 1]



    # --- 邊緣拖拉調整大小 + 視窗拖曳移動 ---

    def _get_resize_direction(self, pos):
        """根據滑鼠位置判斷是否在視窗邊緣，回傳拖拉方向字串或 None"""
        m = self.RESIZE_MARGIN
        rect = self.rect()
        x, y = pos.x(), pos.y()
        w, h = rect.width(), rect.height()

        on_left = x <= m
        on_right = x >= w - m
        on_top = y <= m
        on_bottom = y >= h - m

        if on_top and on_left:
            return 'top_left'
        if on_top and on_right:
            return 'top_right'
        if on_bottom and on_left:
            return 'bottom_left'
        if on_bottom and on_right:
            return 'bottom_right'
        if on_left:
            return 'left'
        if on_right:
            return 'right'
        if on_top:
            return 'top'
        if on_bottom:
            return 'bottom'
        return None

    def _update_cursor_shape(self, direction):
        """根據拖拉方向更新滑鼠游標樣式"""
        if direction == self._last_cursor_direction:
            return

        cursor_map = {
            'left': Qt.CursorShape.SizeHorCursor,
            'right': Qt.CursorShape.SizeHorCursor,
            'top': Qt.CursorShape.SizeVerCursor,
            'bottom': Qt.CursorShape.SizeVerCursor,
            'top_left': Qt.CursorShape.SizeFDiagCursor,
            'bottom_right': Qt.CursorShape.SizeFDiagCursor,
            'top_right': Qt.CursorShape.SizeBDiagCursor,
            'bottom_left': Qt.CursorShape.SizeBDiagCursor,
        }
        if direction and direction in cursor_map:
            self.setCursor(QCursor(cursor_map[direction]))
        else:
            self.unsetCursor()
        self._last_cursor_direction = direction

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            direction = self._get_resize_direction(event.position().toPoint())
            if direction:
                # 開始邊緣拖拉調整大小
                self._resize_direction = direction
                self._resize_origin = event.globalPosition().toPoint()
                self._resize_geometry = self.geometry()
                self.drag_position = None
            else:
                # 開始視窗拖曳移動
                self._resize_direction = None
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.NoButton:
            # 滑鼠未按下：僅更新游標樣式
            direction = self._get_resize_direction(event.position().toPoint())
            self._update_cursor_shape(direction)
            event.accept()
            return

        if event.buttons() == Qt.MouseButton.LeftButton:
            if self._resize_direction:
                # 正在邊緣拖拉調整大小
                global_pos = event.globalPosition().toPoint()
                diff = global_pos - self._resize_origin
                geo = self._resize_geometry
                min_w = self.minimumWidth()
                min_h = self.minimumHeight()

                d = self._resize_direction
                new_x, new_y = geo.x(), geo.y()
                new_w, new_h = geo.width(), geo.height()

                if 'right' in d:
                    new_w = max(min_w, geo.width() + diff.x())
                if 'bottom' in d:
                    new_h = max(min_h, geo.height() + diff.y())
                if 'left' in d:
                    delta = min(diff.x(), geo.width() - min_w)
                    new_x = geo.x() + delta
                    new_w = geo.width() - delta
                if 'top' in d:
                    delta = min(diff.y(), geo.height() - min_h)
                    new_y = geo.y() + delta
                    new_h = geo.height() - delta

                self.setGeometry(new_x, new_y, new_w, new_h)
                event.accept()
            elif self.drag_position is not None:
                # 正在拖曳移動視窗
                self.move(event.globalPosition().toPoint() - self.drag_position)
                event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_position = None
        self._resize_direction = None
        self._resize_origin = None
        self._resize_geometry = None
        # 釋放後更新游標
        direction = self._get_resize_direction(event.position().toPoint())
        self._update_cursor_shape(direction)
        event.accept()

    def load_settings(self):
        """從 SettingsManager 載入設定並更新 UI"""
        self.settings_manager.load_settings_to_ui()

    def save_settings(self):
        """使用 SettingsManager 保存當前設定"""
        self.settings_manager.save_settings_from_ui()
        
        # 立即應用新的瞄準邏輯參數，無需重啟 YOLO
        if hasattr(self, 'new_aim_logic') and self.new_aim_logic:
            self.new_aim_logic.update_parameters()
            logger.info("Aim logic parameters updated from saved settings.")

    def on_yolo_version_changed(self, text):
        """處理YOLO版本變更事件"""
        version_map = {"請選擇YOLO版本": "none", "YOLOv5": "v5", "YOLOv8": "v8", "YOLOv11": "v11", "YOLOv12": "v12"}
        self.selected_yolo_version = version_map.get(text, "none")
        
        if self.selected_yolo_version in ["v5", "v8", "v11", "v12"]:
            self.statusBar().showMessage(f"YOLO 版本已設定為: {self.selected_yolo_version.upper()}，請保存設定以應用變更", 3000)
        else:
            self.statusBar().showMessage("請選擇一個有效的YOLO版本", 3000)

    def main_processing_loop(self):
        logger.info("Main processing loop started.")
        while not self.exit_event.is_set():
            # FPS 統計移到成功取得推論結果後
            if time.time() - self.start_time >= 1.0:
                self.current_fps = self.frame_count
                self.frame_count = 0
                self.start_time = time.time()
                if self.show_fps_overlay_enabled:
                    self.fps_updated.emit(self.current_fps)
 
            try:
                # 步驟 1：非阻塞地提交新幀（如果有的話）
                if self.new_frame_event.is_set():
                    self.new_frame_event.clear()
                    frame, frame_capture_timestamp = self.capture_manager.get_frame()
                    if frame is not None:
                        self.async_pipeline.submit_frame(frame, frame_capture_timestamp)
                
                # 步驟 2：等待推論結果就緒（主要阻塞點）
                # 使用短超時以便定期檢查新幀和退出事件
                if not self.async_pipeline._result_ready_event.wait(timeout=0.005):
                    # 超時：沒有新結果，回到迴圈頂部檢查新幀和退出事件
                    continue
                
                detection_results, inference_time, result_timestamp = self.async_pipeline.get_results()
                
                # 如果還沒有結果（首幀），跳過此次迴圈
                if detection_results is None:
                    continue
                
                # 成功取得推論結果，計入 FPS（這才是真正的推論完成次數）
                self.frame_count += 1
                
                # 使用結果的時間戳計算延遲（如果有）
                if result_timestamp is not None:
                    frame_capture_timestamp = result_timestamp
                
                preview_active = self.preview_enabled and self.preview_window is not None
                if preview_active:
                    processed_frame = self.frame_manager.get_processed_frame_buffer(frame)
                else:
                    processed_frame = frame
                center_x_screen = self.capture_area["width"] // 2
                center_y_screen = self.capture_area["height"] // 2

                target_to_aim = None
                nms_boxes_xyxy = None
                nms_scores = None
                nms_class_ids = None

                if detection_results[0].size > 0:
                    boxes_xywh, scores, class_ids = detection_results
                    
                    model_input_size = self.inference_manager.input_shape[2]
                    capture_size = self.capture_area["width"]
                    
                    if model_input_size > 0:
                        scale_factor = capture_size / model_input_size
                        if scale_factor != 1.0:
                            boxes_xywh *= scale_factor


                    conf_mask = scores > self.yolo_confidence
                    
                    if self.target_class != "ALL":
                        try:
                            # 支援多選目標類別 (逗號分隔)
                            target_ids = [int(x.strip()) for x in self.target_class.split(",") if x.strip()]
                            if target_ids:
                                class_mask = np.isin(class_ids, target_ids)
                                final_mask = np.logical_and(conf_mask, class_mask)
                            else:
                                final_mask = conf_mask
                        except ValueError:
                            final_mask = conf_mask
                    else:
                        final_mask = conf_mask

                    if np.any(final_mask):
                        valid_boxes = boxes_xywh[final_mask]
                        valid_scores = scores[final_mask]
                        valid_class_ids = class_ids[final_mask]
                        
                        # 轉換為 xyxy 格式供 NMS 使用
                        boxes_xyxy = np.zeros((len(valid_boxes), 4), dtype=np.float32)
                        boxes_xyxy[:, 0] = valid_boxes[:, 0] - valid_boxes[:, 2] / 2  # x1
                        boxes_xyxy[:, 1] = valid_boxes[:, 1] - valid_boxes[:, 3] / 2  # y1
                        boxes_xyxy[:, 2] = valid_boxes[:, 0] + valid_boxes[:, 2] / 2  # x2
                        boxes_xyxy[:, 3] = valid_boxes[:, 1] + valid_boxes[:, 3] / 2  # y2
                        
                        # 應用 NMS 過濾重疊框
                        indices = cv2.dnn.NMSBoxes(
                            boxes_xyxy,
                            valid_scores,
                            self.yolo_confidence,
                            self.iou_threshold
                        )
                        
                        if len(indices) > 0:
                            # 處理 indices 格式（OpenCV 版本差異）
                            if isinstance(indices, np.ndarray):
                                indices = indices.flatten()
                            else:
                                indices = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]
                            
                            nms_boxes_xyxy = boxes_xyxy[indices]
                            nms_scores = valid_scores[indices]
                            nms_class_ids = valid_class_ids[indices]
                            
                            logger.debug(f"Detected {len(indices)} targets after NMS, iou_threshold={self.iou_threshold:.2f}")

                # 儲存 NMS 結果供繪圖使用
                self.nms_detections = (nms_boxes_xyxy, nms_scores, nms_class_ids) if nms_boxes_xyxy is not None else None
                
                if self.nms_detections is not None:
                    det_xyxy, det_scores, det_class_ids = self.nms_detections
                    # 從所有偵測目標中選擇距離螢幕中心最近的目標
                    best_target_idx = 0
                    if len(det_xyxy) > 1:
                        min_distance = float('inf')
                        for i in range(len(det_xyxy)):
                            box = det_xyxy[i]
                            cx = (box[0] + box[2]) / 2
                            cy = (box[1] + box[3]) / 2
                            dist = ((cx - center_x_screen)**2 + (cy - center_y_screen)**2)**0.5
                            if dist < min_distance:
                                min_distance = dist
                                best_target_idx = i
                    
                    box = det_xyxy[best_target_idx]
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    
                    target_distance = ((x_center - center_x_screen)**2 + (y_center - center_y_screen)**2)**0.5
                    
                    self._best_target_idx = best_target_idx
                    target_to_aim = {
                        'center_x': x_center,
                        'center_y': y_center,
                        'box_width': box[2] - box[0],
                        'box_height': box[3] - box[1],
                        'distance': target_distance
                    }

                self.new_aim_logic.process_data(target_to_aim)

                # 計算完整延遲：從實際截圖開始到追蹤完畢（準備調用滑鼠移動）
                # 使用節流控制更新頻率，避免過於頻繁的UI更新影響效能
                if self.show_latency_overlay_enabled and frame_capture_timestamp is not None:
                    current_time = time.perf_counter()
                    # 只有當距離上次更新超過指定間隔時才更新
                    if current_time - self._last_latency_update_time >= self._latency_update_interval:
                        total_latency_ms = (current_time - frame_capture_timestamp) * 1000
                        self.latency_updated.emit(total_latency_ms)
                        self._last_latency_update_time = current_time

                if self.aim_overlay_enabled and self.aim_overlay_window:
                    # 使用 new_aim_logic 計算後的結果來判斷目標是否在瞄準範圍內
                    # 這確保了 overlay 顏色變化與實際鎖定邏輯完全一致
                    self.aim_overlay_window.set_target_in_range(self.new_aim_logic.target_in_aim_range)

                if self.show_detection_boxes and self.nms_detections is not None:
                    # 只繪製當前鎖定的目標（距離中心最近的）
                    idx = getattr(self, '_best_target_idx', 0)
                    det_xyxy = self.nms_detections[0]
                    box = det_xyxy[idx]
                    screen_x1 = (self.capture_area['left'] + box[0]) / self.screen_scaling_factor
                    screen_y1 = (self.capture_area['top'] + box[1]) / self.screen_scaling_factor
                    screen_x2 = (self.capture_area['left'] + box[2]) / self.screen_scaling_factor
                    screen_y2 = (self.capture_area['top'] + box[3]) / self.screen_scaling_factor
                    boxes_to_draw = [(screen_x1, screen_y1, screen_x2, screen_y2)]
                    self._push_latest_boxes_to_draw_queue(boxes_to_draw)
                elif self.show_detection_boxes:
                    self._push_latest_boxes_to_draw_queue([])

                # 只有在可渲染預覽時，才執行預覽繪圖與緩衝保留
                if preview_active:
                    if self.nms_detections is not None:
                        det_xyxy, det_scores, det_class_ids = self.nms_detections
                        best_idx = getattr(self, '_best_target_idx', 0)
                        yolo_ver = getattr(self.inference_manager, 'yolo_version', 'unknown')
                        
                        # 繪製所有偵測到的目標
                        for i in range(len(det_xyxy)):
                            box = det_xyxy[i]
                            confidence = det_scores[i]
                            class_id = det_class_ids[i]
                            
                            x_center = (box[0] + box[2]) / 2
                            y_center = (box[1] + box[3]) / 2
                            distance = sqrt((x_center - center_x_screen)**2 + (y_center - center_y_screen)**2)
                            
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            # 鎖定目標用綠色，其他目標用灰色
                            color = (0, 255, 0) if i == best_idx else (128, 128, 128)
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                            
                            if hasattr(self.inference_manager, 'model_names') and self.inference_manager.model_names:
                                class_name = self.inference_manager.model_names.get(class_id, f"Class_{class_id}")
                            else:
                                class_name = f"Class_{class_id}"
                            
                            label = f"{class_name}: {confidence:.3f} ({distance:.0f}px) [{yolo_ver}]"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            
                            cv2.rectangle(processed_frame, (x1, y1 - label_size[1] - 10),
                                        (x1 + label_size[0], y1), color, -1)
                            
                            text_color = (0, 0, 0) if i == best_idx else (255, 255, 255)
                            cv2.putText(processed_frame, label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)



                    cv2.circle(processed_frame, (center_x_screen, center_y_screen), 5, (0, 0, 255), -1)
                    
                    half_width = self.aim_width // 2
                    half_height = self.aim_height // 2
                    top_left = (center_x_screen - half_width, center_y_screen - half_height)
                    bottom_right = (center_x_screen + half_width, center_y_screen + half_height)
                    cv2.rectangle(processed_frame, top_left, bottom_right, (255, 0, 0), 2)
                    
                    # cv2.circle(processed_frame, (center_x_screen, center_y_screen), self.aim_range, (255, 0, 0), 2)
                    
                    with self.frame_lock:
                        if self.latest_processed_frame is not None:
                            self.frame_manager.return_frame_buffer(self.latest_processed_frame)
                        self.latest_processed_frame = processed_frame
                else:
                    # 非預覽狀態下，立即回收當前處理幀，避免 buffer pool 無法重用
                    self.frame_manager.return_frame_buffer(processed_frame)

            except Exception as e:
                logger.error(f"Error in main processing loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(0.1)
        logger.info("Main processing loop finished.")
        
        self.frame_manager.cleanup()
        
        if hasattr(self, 'thread_pool_manager'):
            mouse_stats = self.thread_pool_manager.get_mouse_pool_stats()
            logger.info(f"Thread pool mouse stats: {mouse_stats}")
        


    def toggle_yolo(self):
        if self.yolo_enabled:


            self.yolo_enabled = False
            
            self.exit_event.set()
            self.video_timer.stop()
            self.aim_scale_timer.stop()
            # 確保停止時恢復原始瞄準範圍
            if self.is_zoomed:
                self.aim_dimensions_changed.emit(self.original_aim_width, self.original_aim_height)
                self.is_zoomed = False
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1)

            
            if self.async_pipeline:
                self.async_pipeline.stop()
                self.async_pipeline = None
            
            if self.capture_manager:
                self.capture_manager.stop()

            if self.inference_manager:
                self.inference_manager.cleanup()

            self.toggle_yolo_button.setText("啟動 YOLO")
            if self.preview_window:
                self.preview_window.video_label.setText("預覽已停止")
                self.preview_window.hide()
            self.drawing_timer.stop()
            logger.info("YOLO pipeline stopped.")
            self.statusBar().showMessage("YOLO 已停止", 3000)
        else:
            self.statusBar().showMessage("正在初始化...", 0)
            QApplication.processEvents()

            selected_version = getattr(self, 'selected_yolo_version', 'none')
            if selected_version not in ['v5', 'v8', 'v11', 'v12']:
                QMessageBox.warning(self, "提示", "請先在下拉選單中選擇有效的YOLO版本 (v5、v8、v11或v12) 再啟動。")
                self.statusBar().showMessage("請選擇YOLO版本", 5000)
                return

            self.statusBar().showMessage("正在初始化...", 0)
            QApplication.processEvents()

            try:
                self.inference_manager = EnhancedInferenceManager(self, yolo_version=selected_version, image_processor=self.image_processor)
                if not self.inference_manager.initialize():
                    QMessageBox.critical(self, "錯誤", "推理引擎初始化失敗（ONNX/TensorRT），請檢查日誌。")
                    self.inference_manager = None
                    self.statusBar().showMessage("初始化失敗!", 5000)
                    return
            except Exception as e:
                logger.error(f"創建InferenceManager時出錯: {e}")
                QMessageBox.critical(self, "錯誤", f"初始化失敗: {e}")
                self.statusBar().showMessage("初始化失敗!", 5000)
                return

            self.model_size_input.setText(str(self.model_size))
            self.update_video_label_size()
            if self.preview_enabled and self.preview_window:
                self.preview_window.show()
            self.capture_area = self.calculate_capture_area()
            
            if hasattr(self, 'new_aim_logic') and self.new_aim_logic:
                self.new_aim_logic.update_parameters()

            self.exit_event.clear()

            selected_monitor = self.get_monitor_by_index(getattr(self, 'capture_monitor_index', 1))
            
            self.capture_manager = ScreenCaptureManager(
                self.capture_area, 
                self.exit_event, 
                self.new_frame_event, 
                self.capture_fps,
                self.capture_source,
                self.obs_ip,
                self.obs_port,
                capture_monitor_index=self.capture_monitor_index,
                capture_monitor_bounds=selected_monitor,
            )
            self.capture_manager.start()

            # 初始化非同步推論管道
            self.async_pipeline = AsyncInferencePipeline(self.inference_manager, self.exit_event)
            self.async_pipeline.start()

            self.processing_thread = threading.Thread(target=self.main_processing_loop, daemon=True)
            self.processing_thread.start()



            self.yolo_enabled = True
            


            # 動態設定計時器更新率以匹配截圖速率
            update_interval = self.get_ui_refresh_interval(self.capture_fps)
            self.video_timer.start(update_interval)
            self.drawing_timer.start(update_interval)
            self.aim_scale_timer.start(50) # 每 50 毫秒檢查一次縮放狀態
            
            # 初始化目標類別選單
            if hasattr(self.inference_manager, 'model_names'):
                self.ui_handler.setup_target_class_menu(self.inference_manager.model_names)
            
            self.toggle_yolo_button.setText("停止 YOLO")
            logger.info("YOLO pipeline started.")
            self.statusBar().showMessage("YOLO 已啟動", 3000)


    def toggle_preview_window(self):
        self.preview_enabled = not self.preview_enabled
        
        # 如果UI視窗可見，則直接切換預覽視窗的顯示狀態
        if not self.isHidden() and self.preview_window:
            if self.preview_enabled and self.yolo_enabled:
                self.preview_window.show()
            else:
                self.preview_window.hide()
        
        # 如果UI視窗是隱藏的，這個切換只會影響背景處理邏輯(是否繪製幀)
        logger.info(f"預覽功能已 {'啟用' if self.preview_enabled else '禁用'}")
        self.toggle_preview_button.setText("關閉 YOLO 預覽" if self.preview_enabled else "開啟 YOLO 預覽")

        # 關閉預覽時，立即釋放最後一幀，避免殘留記憶體占用
        if not self.preview_enabled:
            with self.frame_lock:
                if self.latest_processed_frame is not None:
                    self.frame_manager.return_frame_buffer(self.latest_processed_frame)
                    self.latest_processed_frame = None



    def update_video_label_size(self):
        """根據模型大小動態調整預覽視窗中影像標籤的大小"""
        if self.preview_window:
            self.preview_window.video_label.setFixedSize(self.model_size, self.model_size)
            self.preview_window.resize(self.model_size, self.model_size)


    def update_aim_dimensions_ui(self, width, height):
        self.aim_width = width
        self.aim_height = height
        
        if hasattr(self, 'aim_width_slider'):
            self.aim_width_slider.blockSignals(True)
            self.aim_width_slider.setValue(self.aim_width)
            self.aim_width_slider.blockSignals(False)
            
        if hasattr(self, 'aim_height_slider'):
            self.aim_height_slider.blockSignals(True)
            self.aim_height_slider.setValue(self.aim_height)
            self.aim_height_slider.blockSignals(False)
            
        if self.aim_overlay_window:
            self.aim_overlay_window.update_size(self.aim_width, self.aim_height)
  
    def update_visual_offset_ui(self, new_offset_x):
        self.offset_centerx_slider.blockSignals(True)
        self.offset_centerx_slider.setValue(int(new_offset_x * 100))
        self.offset_centerx_slider.blockSignals(False)
        self.offset_centerx_value_label.setText(f"{new_offset_x:.2f}")

    def _push_latest_boxes_to_draw_queue(self, boxes):
        """只保留最新一筆框資料，避免 queue 堆積造成延遲與額外負擔。"""
        if not hasattr(self, 'boxes_to_draw_queue') or self.boxes_to_draw_queue is None:
            return

        try:
            while True:
                self.boxes_to_draw_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.boxes_to_draw_queue.put_nowait(boxes)
        except Exception as e:
            logger.debug(f"Failed to enqueue latest boxes: {e}")

    def update_drawing_overlay(self):
        if not self.show_detection_boxes or not self.draw_screen_window:
            return
        
        try:
            latest_boxes = None
            while True:
                try:
                    latest_boxes = self.boxes_to_draw_queue.get_nowait()
                except queue.Empty:
                    break

            if latest_boxes is not None:
                self.draw_screen_window.update_boxes(latest_boxes)
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error updating drawing overlay: {e}")
    
    def check_thread_pool_health(self):
        try:
            if hasattr(self, 'thread_pool_manager'):
                if not self.thread_pool_manager.is_healthy():
                    logger.warning("Thread pool unhealthy, attempting restart...")
                    self.thread_pool_manager.restart_if_needed()
                
                if hasattr(self, '_health_check_count'):
                    self._health_check_count += 1
                else:
                    self._health_check_count = 1
                
                if self._health_check_count % 10 == 0:
                    stats = self.thread_pool_manager.get_mouse_pool_stats()
                    logger.info(f"Thread pool periodic stats: {stats}")
                    self.thread_pool_manager.reset_stats()
                    

                    
                    
        except Exception as e:
            logger.error(f"Error in thread pool health check: {e}")
    



    def calculate_capture_area(self):
        monitor = self.get_monitor_by_index(getattr(self, 'capture_monitor_index', 1))

        capture_width = min(self.model_size, monitor['width'])
        capture_height = min(self.model_size, monitor['height'])

        if capture_width != self.model_size or capture_height != self.model_size:
            logger.warning(
                f"模型尺寸 {self.model_size} 超出顯示器解析度，截圖區域已調整為 {capture_width}x{capture_height}"
            )

        left = monitor['left'] + (monitor['width'] - capture_width) // 2
        top = monitor['top'] + (monitor['height'] - capture_height) // 2
        return {
            "top": top,
            "left": left,
            "width": capture_width,
            "height": capture_height
        }

    def update_display(self):
        if not (self.yolo_enabled and self.preview_enabled and self.preview_window):
            return

        with self.frame_lock:
            if self.latest_processed_frame is None:
                return
            frame_to_display = self.frame_manager.get_display_frame_buffer(self.latest_processed_frame)

        try:
            fps_text = f"FPS: {self.current_fps}"
            cv2.putText(frame_to_display, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 預覽用：BGR→RGB 轉換以配合 QImage Format_RGB888
            rgb_frame = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.preview_window.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_window.video_label.setPixmap(scaled_pixmap)
        finally:
            # 只歸還從 pool 取得的原始 BGR 緩衝，避免錯誤回收 cvtColor 新陣列
            self.frame_manager.return_frame_buffer(frame_to_display)


    def check_aim_scale(self):
        """使用 QTimer 定期檢查滑鼠右鍵狀態以實現自動縮放。"""
        if not self.auto_scale_aim_range:
            if self.is_zoomed:
                self.aim_dimensions_changed.emit(self.original_aim_width, self.original_aim_height)
                self.is_zoomed = False
            return
    
        # VK_RBUTTON 的虛擬鍵碼是 0x02
        right_button_pressed = keyboard_listener.is_pressed(0x02)
    
        if right_button_pressed and not self.is_zoomed:
            new_width = int(self.original_aim_width / (self.auto_scale_factor + 1))
            new_height = int(self.original_aim_height / (self.auto_scale_factor + 1))
            self.aim_dimensions_changed.emit(new_width, new_height)
            
            self.is_zoomed = True
            logger.debug(f"Zoom in signal emitted: dimensions set to {new_width}x{new_height}")
        elif not right_button_pressed and self.is_zoomed:
            self.aim_dimensions_changed.emit(self.original_aim_width, self.original_aim_height)
            self.is_zoomed = False
            logger.debug(f"Zoom out signal emitted: dimensions restored to {self.original_aim_width}x{self.original_aim_height}")
    
    def toggle_fps_overlay(self, state):
        self.show_fps_overlay_enabled = bool(state)
        if self.show_fps_overlay_enabled:
            if self.fps_overlay_window:
                self.fps_overlay_window.show()
            logger.info("FPS 已啟用。")
        else:
            if self.fps_overlay_window:
                self.fps_overlay_window.hide()
            logger.info("FPS 已停用。")

    def toggle_latency_overlay(self, state):
        self.show_latency_overlay_enabled = bool(state)
        if self.show_latency_overlay_enabled:
            # 啟用時：重置節流計時器，讓第一次更新立即發生
            self._last_latency_update_time = 0
            if self.latency_overlay_window:
                self.latency_overlay_window.clear_history()
                self.latency_overlay_window.show()
            logger.info("延遲覆蓋層已啟用。")
        else:
            # 關閉時：隱藏視窗，清除歷史記錄，確保無額外開銷
            if self.latency_overlay_window:
                self.latency_overlay_window.hide()
                self.latency_overlay_window.clear_history()
            logger.info("延遲覆蓋層已停用。")

    def update_recoil_control(self):
        self.recoil_control.set_config(
            enabled=self.recoil_control_enabled,
            x_strength=self.recoil_x_strength,
            y_strength=self.recoil_y_strength,
            delay=self.recoil_delay,
            mouse_move_mode=self.mouse_move_mode,
            trigger_keys_str=self.recoil_trigger_keys
        )

    def fully_close(self):
        """完全關閉應用程式並釋放所有資源"""
        logger.info("執行完全關閉程序...")
        self.exit_event.set()
        if self.yolo_enabled:
            self.toggle_yolo()

        if self.aim_overlay_window:
            self.aim_overlay_window.close()
        if self.fps_overlay_window:
            self.fps_overlay_window.close()
        if self.latency_overlay_window:
            self.latency_overlay_window.close()
        if self.draw_screen_window:
            self.draw_screen_window.close()
        if self.preview_window:
            self.preview_window.close()
            


        # 確保所有計時器都已停止
        self.video_timer.stop()
        self.drawing_timer.stop()
        self.thread_pool_health_timer.stop()
        self.aim_scale_timer.stop()
        self.stop_aimbot_hotkey_listener()

        QApplication.instance().quit()
        logger.info("應用程式已完全關閉。")

    def closeEvent(self, event):
        """處理視窗關閉事件，將其委派給系統匣管理員"""
        if hasattr(self, 'tray_manager'):
            self.tray_manager.handle_close_event(event)
        else:
            # 若系統匣管理員不存在，則執行預設的關閉行為
            super().closeEvent(event)

    def showEvent(self, event):
        """處理視窗顯示事件，重新建立預覽視窗"""
        super().showEvent(event)
        if not hasattr(self, 'preview_window') or not self.preview_window:
            self.preview_window = PreviewWindow()
            self.update_video_label_size()
            logger.info("預覽視窗已重新建立。")
        
        if self.preview_enabled and self.yolo_enabled:
            self.preview_window.show()

        # 如果 YOLO 與預覽都在運行，才啟動預覽更新計時器
        if self.yolo_enabled and self.preview_enabled and self.preview_window and not self.video_timer.isActive():
            update_interval = self.get_ui_refresh_interval(self.capture_fps)
            self.video_timer.start(update_interval)
            logger.info(f"UI顯示，預覽更新計時器已重新啟動 (間隔: {update_interval}ms)。")

    def hideEvent(self, event):
        """處理視窗隱藏事件，關閉並釋放預覽視窗資源，並停止UI更新"""
        # 這是由 tray_manager.handle_close_event 間接呼叫的
        
        # 1. 停止觸發UI更新的計時器 (節省CPU)
        if self.video_timer.isActive():
            self.video_timer.stop()
            logger.info("UI隱藏，預覽更新計時器已停止。")

        # 2. 關閉並釋放預覽視窗 (節省記憶體)
        if self.preview_window:
            if hasattr(self.preview_window, 'video_label'):
                self.preview_window.video_label.clear()
                self.preview_window.video_label.setText("YOLO 預覽")
            self.preview_window.close()
            self.preview_window = None # 釋放參考
            logger.info("預覽視窗已關閉並釋放資源。")
            
        # 3. 釋放最後一幀的記憶體
        with self.frame_lock:
            if self.latest_processed_frame is not None:
                self.frame_manager.return_frame_buffer(self.latest_processed_frame)
                self.latest_processed_frame = None
                logger.info("最後一幀預覽影像已釋放。")
            
        super().hideEvent(event)

    def set_process_priority(self):
       """根據設定設定處理程序優先權"""
       try:
           p = psutil.Process(os.getpid())
           priority_map = {
               "即時": psutil.REALTIME_PRIORITY_CLASS,
               "高": psutil.HIGH_PRIORITY_CLASS,
               "高於標準": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
               "標準": psutil.NORMAL_PRIORITY_CLASS,
               "低於標準": psutil.BELOW_NORMAL_PRIORITY_CLASS,
               "低": psutil.IDLE_PRIORITY_CLASS
           }
           
           desired_priority_class = priority_map.get(self.process_priority, psutil.NORMAL_PRIORITY_CLASS)
           
           current_priority_before = p.nice()
           
           p.nice(desired_priority_class)
           
           current_priority_after = p.nice()

           if current_priority_after == desired_priority_class:
               logger.info(f"處理程序優先權成功設定為: {self.process_priority} ({desired_priority_class})")
           else:
               logger.warning(f"嘗試設定優先權為 {self.process_priority} ({desired_priority_class})，但目前為 ({current_priority_after})。")
               if desired_priority_class > psutil.NORMAL_PRIORITY_CLASS:
                   self.statusBar().showMessage("設定高優先權失敗，請嘗試以系統管理員身分執行。", 7000)
               else:
                    self.statusBar().showMessage(f"優先權設定為: {self.process_priority}", 5000)

       except psutil.AccessDenied:
           logger.error("設定處理程序優先權失敗: 存取被拒。請嘗試以系統管理員身分執行。")
           self.statusBar().showMessage("設定優先權失敗: 存取被拒。請以系統管理員身分執行。", 7000)
       except Exception as e:
           logger.error(f"設定處理程序優先權時發生未知錯誤: {e}")

if __name__ == "__main__":
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    
    # 啟動全域鍵盤監聽器
    keyboard_listener.start()
    
    # 註冊程式退出時的清理函式
    def cleanup():
        logger.info("Application is shutting down. Cleaning up resources...")
        keyboard_listener.stop()
        shutdown_global_thread_pool()
        logger.info("Cleanup finished.")
    
    atexit.register(cleanup)
    
    app.setQuitOnLastWindowClosed(False)
    
    window = main()
    window.show()
    sys.exit(app.exec())
