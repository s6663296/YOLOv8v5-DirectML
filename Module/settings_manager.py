from .config import Config
from .logger import logger
import os
from .utils import resource_path

class SettingsManager:
    """處理所有應用程式設定的載入、儲存和更新。"""
    def __init__(self, main_window):
        """
        初始化設定管理器。

        :param main_window: 主視窗 (SudaneseboyApp) 的實例。
        """
        self.window = main_window

    def load_initial_settings(self):
        """
        在應用程式啟動時載入初始設定值到主視窗的屬性中。
        """
        # --- 模型與偵測 ---
        self.window.yolo_confidence = Config.get("confidence", 0.3)
        self.window.model_size = Config.get("model_size", 320)
        self.window.target_class = Config.get("target_class", "0")
        self.window.iou_threshold = Config.get("iou_threshold", 0.45)
        self.window.selected_yolo_version = Config.get("yolo_version", "none")

        # --- 瞄準設定 ---
        self.window.offset_centerx = Config.get("offset_centerx", 0.0)
        self.window.offset_centery = Config.get("offset_centery", 0.0)
        self.window.aim_range = Config.get("aim_range", 150)
        self.window.original_aim_range = self.window.aim_range
        self.window.aim_speed_x = Config.get("aim_speed_x", 6.7)
        self.window.aim_speed_y = Config.get("aim_speed_y", 8.3)
        self.window.aimbot_enabled = Config.get("aimBot", True)
        self.window.lock_key = Config.get("lockKey", "VK_RBUTTON")
        self.window.auto_lock_mode = not self.window.lock_key
        self.window.side_key_lock_enabled = Config.get("side_key_lock_enabled", False)
        self.window.side_lock_key = Config.get("side_lock_key", "VK_XBUTTON1")
        self.window.mouse_move_mode = Config.get("mouse_move_mode", "按鍵魔盒-鍵鼠轉接器")

        # --- 預測設定 ---
        self.window.prediction_enabled = Config.get("prediction", False)
        self.window.prediction_offset = Config.get("prediction_offset", 0.03)
        self.window.prediction_threshold = Config.get("prediction_threshold", 1)
        self.window.prediction_time = Config.get("prediction_time", 1.0)
        self.window.prediction_delay = Config.get("prediction_delay", 1.0)

        # --- 範圍縮放 ---
        self.window.auto_scale_aim_range = Config.get("auto_scale_aim_range", False)
        self.window.auto_scale_factor = Config.get("auto_scale_factor", 0.0)

        # --- 後座力控制 ---
        self.window.recoil_control_enabled = Config.get("recoil_control_enabled", False)
        self.window.recoil_x_strength = Config.get("recoil_x_strength", 0)
        self.window.recoil_y_strength = Config.get("recoil_y_strength", 5)
        self.window.recoil_delay = Config.get("recoil_delay", 10)
        self.window.recoil_trigger_keys = Config.get("recoil_trigger_keys", "VK_LBUTTON")

        # --- 疊加層與顯示 ---
        overlay_color_values = [
            Config.get("overlay_color_r", 0),
            Config.get("overlay_color_g", 255),
            Config.get("overlay_color_b", 255),
            Config.get("overlay_color_a", 200)
        ]
        self.window.overlay_color = tuple(c[0] if isinstance(c, list) else c for c in overlay_color_values)

        overlay_lock_color_values = [
            Config.get("overlay_lock_color_r", 255),
            Config.get("overlay_lock_color_g", 0),
            Config.get("overlay_lock_color_b", 0),
            Config.get("overlay_lock_color_a", 200)
        ]
        self.window.overlay_lock_color = tuple(c[0] if isinstance(c, list) else c for c in overlay_lock_color_values)
        self.window.show_fps_overlay_enabled = Config.get("show_fps_overlay", False)
        self.window.show_detection_boxes = Config.get("show_detection_boxes", False)

        # --- 系統設定 ---
        self.window.capture_fps = Config.get("capture_fps", 240)
        self.window.mouse_threading = Config.get("mouse_threading", True)

    def load_settings_to_ui(self):
        """從 Config 載入設定並更新 UI。"""
        win = self.window
        
        self.load_initial_settings()

        # --- 模型與偵測 ---
        win.yolo_confidence_slider.setValue(int(win.yolo_confidence * 100))
        win.yolo_confidence_value_label.setText(f"{win.yolo_confidence:.2f}")
        win.model_size_input.setText(str(win.model_size))
        win.detection_target_combobox.setCurrentText(win.target_class)
        win.iou_threshold_slider.setValue(int(win.iou_threshold * 100))
        win.iou_threshold_value_label.setText(f"{win.iou_threshold:.2f}")
        win.model_path_display.setText(Config.get("model_file", "yolov8n.pt"))

        # --- YOLO 版本 ---
        version_map = {"none": "請選擇YOLO版本", "v5": "YOLOv5", "v8": "YOLOv8", "auto": "請選擇YOLO版本"}
        current_text = version_map.get(win.selected_yolo_version, "請選擇YOLO版本")
        win.yolo_version_combobox.blockSignals(True)
        win.yolo_version_combobox.setCurrentText(current_text)
        win.yolo_version_combobox.blockSignals(False)

        # --- 瞄準設定 ---
        win.aimbot_checkbox.setChecked(win.aimbot_enabled)
        win.offset_centerx_slider.setValue(int(win.offset_centerx * 100))
        win.offset_centerx_value_label.setText(f"{win.offset_centerx:.2f}")
        centery_slider_value = max(-100, min(100, int(win.offset_centery * 100)))
        win.offset_centery_slider.setValue(centery_slider_value)
        win.offset_centery_value_label.setText(f"{win.offset_centery:.2f}")
        win.aim_range_slider.setValue(win.aim_range)
        win.aim_range_value_label.setText(str(win.aim_range))
        win.aim_speed_x_slider.setValue(int(win.aim_speed_x * 10))
        win.aim_speed_x_value_label.setText(f"{win.aim_speed_x:.1f}")
        win.aim_speed_y_slider.setValue(int(win.aim_speed_y * 10))
        win.aim_speed_y_value_label.setText(f"{win.aim_speed_y:.1f}")
        win.lock_key_input.setText(win.lock_key)
        win.side_key_lock_enabled_checkbox.setChecked(win.side_key_lock_enabled)
        win.side_lock_key_input.setText(win.side_lock_key)
        win.mouse_move_mode_combobox.setCurrentText(win.mouse_move_mode)

        # --- 預測設定 ---
        win.prediction_checkbox.setChecked(win.prediction_enabled)
        win.prediction_time_slider.setValue(int(win.prediction_time * 10))
        win.prediction_time_value_label.setText(f"{win.prediction_time:.1f}")

        # --- 範圍縮放 ---
        win.auto_scale_aim_range_checkbox.setChecked(win.auto_scale_aim_range)
        win.auto_scale_factor_slider.setValue(int(win.auto_scale_factor * 10))
        win.auto_scale_factor_value_label.setText(f"{win.auto_scale_factor:.1f}")
        win.auto_scale_factor_slider.setEnabled(win.auto_scale_aim_range)
        win.auto_scale_factor_label.setEnabled(win.auto_scale_aim_range)

        # --- 後座力控制 ---
        win.recoil_control_checkbox.setChecked(win.recoil_control_enabled)
        win.recoil_x_strength_slider.setValue(win.recoil_x_strength)
        win.recoil_x_strength_value_label.setText(str(win.recoil_x_strength))
        win.recoil_y_strength_slider.setValue(win.recoil_y_strength)
        win.recoil_y_strength_value_label.setText(str(win.recoil_y_strength))
        win.recoil_delay_slider.setValue(win.recoil_delay)
        win.recoil_delay_value_label.setText(str(win.recoil_delay))
        win.recoil_trigger_keys_input.setText(win.recoil_trigger_keys)
        win.recoil_x_strength_slider.setEnabled(win.recoil_control_enabled)
        win.recoil_y_strength_slider.setEnabled(win.recoil_control_enabled)
        win.recoil_delay_slider.setEnabled(win.recoil_control_enabled)
        win.update_recoil_control()

        # --- 疊加層與顯示 ---
        win.ui_handler.update_color_previews()
        win.show_fps_overlay_checkbox.setChecked(win.show_fps_overlay_enabled)
        win.toggle_fps_overlay(win.show_fps_overlay_enabled)
        win.show_detection_boxes_checkbox.setChecked(win.show_detection_boxes)
        win.ui_handler.on_show_detection_boxes_changed(win.show_detection_boxes)
        win.toggle_preview_button.setText("關閉 YOLO 預覽" if win.preview_enabled else "開啟 YOLO 預覽")

        # --- 系統設定 ---
        win.capture_fps_slider.setValue(win.capture_fps)
        win.capture_fps_value_label.setText(str(win.capture_fps))
        if hasattr(win, 'mouse_thread_checkbox'):
            win.mouse_thread_checkbox.setChecked(win.mouse_threading)
        
        win.update_video_label_size()
        self.update_config_display()

    def save_settings_from_ui(self):
        """從 UI 收集當前設定並保存到 Config。"""
        win = self.window
        
        # --- 模型與偵測 ---
        Config.content["confidence"] = win.yolo_confidence
        Config.content["model_size"] = win.model_size
        Config.content["target_class"] = win.target_class
        Config.content["iou_threshold"] = win.iou_threshold
        Config.content["yolo_version"] = win.selected_yolo_version
        Config.content["model_file"] = win.model_path_display.text()

        # --- 瞄準設定 ---
        Config.content["aimBot"] = win.aimbot_enabled
        Config.content["offset_centerx"] = win.offset_centerx
        Config.content["offset_centery"] = win.offset_centery
        Config.content["aim_range"] = win.aim_range
        Config.content["aim_speed_x"] = win.aim_speed_x
        Config.content["aim_speed_y"] = win.aim_speed_y
        Config.content["lockKey"] = win.lock_key
        Config.content["side_key_lock_enabled"] = win.side_key_lock_enabled
        Config.content["side_lock_key"] = win.side_lock_key
        Config.content["mouse_move_mode"] = win.mouse_move_mode

        # --- 預測設定 ---
        Config.content["prediction"] = win.prediction_enabled
        Config.content["prediction_delay"] = win.prediction_delay
        Config.content["prediction_offset"] = win.prediction_offset
        Config.content["prediction_threshold"] = win.prediction_threshold
        Config.content["prediction_time"] = win.prediction_time

        # --- 範圍縮放 ---
        Config.content["auto_scale_aim_range"] = win.auto_scale_aim_range
        Config.content["auto_scale_factor"] = win.auto_scale_factor

        # --- 後座力控制 ---
        Config.content["recoil_control_enabled"] = win.recoil_control_enabled
        Config.content["recoil_x_strength"] = win.recoil_x_strength
        Config.content["recoil_y_strength"] = win.recoil_y_strength
        Config.content["recoil_delay"] = win.recoil_delay
        Config.content["recoil_trigger_keys"] = win.recoil_trigger_keys

        # --- 疊加層與顯示 ---
        Config.content["overlay_color_r"] = win.overlay_color[0]
        Config.content["overlay_color_g"] = win.overlay_color[1]
        Config.content["overlay_color_b"] = win.overlay_color[2]
        Config.content["overlay_color_a"] = win.overlay_color[3]
        Config.content["overlay_lock_color_r"] = win.overlay_lock_color[0]
        Config.content["overlay_lock_color_g"] = win.overlay_lock_color[1]
        Config.content["overlay_lock_color_b"] = win.overlay_lock_color[2]
        Config.content["overlay_lock_color_a"] = win.overlay_lock_color[3]
        Config.content["show_fps_overlay"] = win.show_fps_overlay_enabled
        Config.content["show_detection_boxes"] = win.show_detection_boxes

        # --- 系統設定 ---
        Config.content["capture_fps"] = win.capture_fps
        Config.content["mouse_threading"] = win.mouse_threading
        
        Config.save_to_current_config()
        win.statusBar().showMessage("設定已儲存", 3000)

    def update_config_display(self):
        """更新顯示當前設定檔名稱的UI"""
        current_config = Config.get_current_config_file()
        if current_config == "settings.json":
            self.window.config_file_display.setText("預設設定")
        else:
            config_name = os.path.splitext(os.path.basename(current_config))
            self.window.config_file_display.setText(f"{config_name} 配置")
