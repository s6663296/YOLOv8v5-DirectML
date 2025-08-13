import win32api
import win32con
import json
import os
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QLabel
from PyQt6.QtCore import Qt
from Module.aim_overlay import AimOverlayWindow
from Module.config import Config
from Module.logger import logger
import Module.control as control
from PyQt6.QtWidgets import QColorDialog
from PyQt6.QtGui import QColor

class UIHandler:
    def __init__(self, main_window):
        self.main_window = main_window

    def init_signals(self):
        """連接從 .ui 檔案載入的元件的信號"""
        win = self.main_window
        win.choose_model_button.clicked.connect(self.choose_model)
        win.aimbot_checkbox.stateChanged.connect(self.on_aimbot_checkbox_state_changed)
        win.offset_centerx_slider.valueChanged.connect(self.on_offset_centerx_slider_value_changed)
        win.offset_centery_slider.valueChanged.connect(self.on_offset_centery_slider_value_changed)
        win.aim_range_slider.valueChanged.connect(self.on_aim_range_slider_value_changed)
        win.aim_speed_x_slider.valueChanged.connect(self.on_aim_speed_x_slider_value_changed)
        win.aim_speed_y_slider.valueChanged.connect(self.on_aim_speed_y_slider_value_changed)
        win.yolo_confidence_slider.valueChanged.connect(self.on_yolo_confidence_slider_value_changed)
        win.model_size_input.textChanged.connect(self.on_model_size_changed)
        win.detection_target_combobox.currentTextChanged.connect(self.on_detection_target_changed)
        win.lock_key_input.textChanged.connect(self.on_lock_key_changed)
        win.toggle_yolo_button.clicked.connect(self.main_window.toggle_yolo)
        win.toggle_aim_overlay_button.clicked.connect(self.toggle_aim_overlay)
        win.side_key_lock_enabled_checkbox.stateChanged.connect(self.on_side_key_lock_enabled_changed)
        win.side_lock_key_input.textChanged.connect(self.on_side_lock_key_changed)
        win.prediction_checkbox.stateChanged.connect(self.on_prediction_checkbox_state_changed)
        win.auto_scale_aim_range_checkbox.stateChanged.connect(self.on_auto_scale_aim_range_changed)
        win.auto_scale_factor_slider.valueChanged.connect(self.on_auto_scale_factor_changed)
        win.prediction_time_slider.valueChanged.connect(self.on_prediction_time_slider_value_changed)
        win.mouse_move_mode_combobox.currentTextChanged.connect(self.on_mouse_move_mode_changed)
        if hasattr(win, 'mouse_thread_checkbox'):
            win.mouse_thread_checkbox.stateChanged.connect(self.on_mouse_thread_changed)
        else:
            logger.warning("UI中找不到 'mouse_thread_checkbox' 元件，滑鼠多執行緒選項將不可用。")
        win.aim_range_changed.connect(win.update_aim_range_ui)
        win.visual_offset_changed.connect(win.update_visual_offset_ui)
        # 後座力控制信號
        win.recoil_control_checkbox.stateChanged.connect(self.on_recoil_control_changed)
        win.recoil_x_strength_slider.valueChanged.connect(self.on_recoil_x_strength_changed)
        win.recoil_y_strength_slider.valueChanged.connect(self.on_recoil_y_strength_changed)
        win.recoil_delay_slider.valueChanged.connect(self.on_recoil_delay_changed)
        win.recoil_trigger_keys_input.textChanged.connect(self.on_recoil_trigger_keys_changed)

        # 顏色設定信號
        if hasattr(win, 'choose_overlay_color_button'):
            win.choose_overlay_color_button.clicked.connect(self.choose_overlay_color)
        else:
            logger.warning("UI中找不到 'choose_overlay_color_button' 元件。")
        if hasattr(win, 'choose_lock_color_button'):
            win.choose_lock_color_button.clicked.connect(self.choose_lock_color)
        else:
            logger.warning("UI中找不到 'choose_lock_color_button' 元件。")


        if hasattr(win, 'show_fps_overlay_checkbox'):
           win.show_fps_overlay_checkbox.stateChanged.connect(self.on_show_fps_overlay_changed)
        else:
           logger.warning("UI中找不到 'show_fps_overlay_checkbox' 元件。")

        if hasattr(win, 'show_detection_boxes_checkbox'):
           win.show_detection_boxes_checkbox.stateChanged.connect(self.on_show_detection_boxes_changed)
        else:
           logger.warning("UI中找不到 'show_detection_boxes_checkbox' 元件。")
           
        if hasattr(win, 'capture_fps_slider'):
           win.capture_fps_slider.valueChanged.connect(self.on_capture_fps_changed)
        else:
           logger.warning("UI中找不到 'capture_fps_slider' 元件。")
           
        if hasattr(win, 'iou_threshold_slider'):
           win.iou_threshold_slider.valueChanged.connect(self.on_iou_threshold_changed)
        else:
           logger.warning("UI中找不到 'iou_threshold_slider' 元件。")

    def choose_model(self):
        win = self.main_window
        file_dialog = QFileDialog(win)
        file_dialog.setNameFilter("YOLO Models (*.pt *.onnx *.engine)")
        file_dialog.setWindowTitle("選擇 YOLO 模型")
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            
            win.model_path_display.setText(selected_file)
            Config.content["model_file"] = selected_file
            
            win.statusBar().showMessage("自訂模型已選擇，請保存設定以應用變更", 5000)
            logger.info(f"用戶選擇自訂模型: {selected_file}")

    def on_aimbot_checkbox_state_changed(self, state):
        win = self.main_window
        win.aimbot_enabled = bool(state)
        Config.content["aimBot"] = win.aimbot_enabled
        status = '啟用' if win.aimbot_enabled else '禁用'
        logger.info(f"瞄準功能 {status}")
        win.statusBar().showMessage(f"瞄準功能 {status}，請保存設定以應用變更", 3000)

    def on_offset_centerx_slider_value_changed(self, value):
        win = self.main_window
        win.offset_centerx = value / 100.0
        win.offset_centerx_value_label.setText(f"{win.offset_centerx:.2f}")
        Config.content["offset_centerx"] = win.offset_centerx
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_offset_centery_slider_value_changed(self, value):
        win = self.main_window
        win.offset_centery = value / 100.0
        win.offset_centery_value_label.setText(f"{win.offset_centery:.2f}")
        Config.content["offset_centery"] = win.offset_centery
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_aim_range_slider_value_changed(self, value):
        win = self.main_window
        win.aim_range = value
        win.original_aim_range = value
        win.aim_range_value_label.setText(str(win.aim_range))
        Config.content["aim_range"] = win.aim_range
        if win.aim_overlay_window:
            win.aim_overlay_window.set_aim_range(win.aim_range)

    def on_aim_speed_x_slider_value_changed(self, value):
        win = self.main_window
        win.aim_speed_x = value / 10.0
        win.aim_speed_x_value_label.setText(f"{win.aim_speed_x:.1f}")
        Config.content["aim_speed_x"] = win.aim_speed_x
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_aim_speed_y_slider_value_changed(self, value):
        win = self.main_window
        win.aim_speed_y = value / 10.0
        win.aim_speed_y_value_label.setText(f"{win.aim_speed_y:.1f}")
        Config.content["aim_speed_y"] = win.aim_speed_y
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_detection_target_changed(self, text):
        win = self.main_window
        win.target_class = text
        Config.content["target_class"] = win.target_class
        logger.info(f"偵測目標類別更改為: {win.target_class}")

    def on_yolo_confidence_slider_value_changed(self, value):
        win = self.main_window
        win.yolo_confidence = value / 100.0
        win.yolo_confidence_value_label.setText(f"{win.yolo_confidence:.2f}")
        Config.content["confidence"] = win.yolo_confidence
        logger.info(f"YOLO 置信度更改為: {win.yolo_confidence:.2f}")

    def on_model_size_changed(self, text):
        win = self.main_window
        try:
            size = int(text)
            if size > 0 and size % 32 == 0:
                win.model_size = size
                Config.content["model_size"] = win.model_size
                logger.info(f"模型輸入尺寸更改為: {win.model_size}")
                win.capture_area = win.calculate_capture_area()
                win.update_video_label_size()
                win.statusBar().showMessage(f"模型尺寸設定為 {win.model_size}", 3000)
            else:
                win.statusBar().showMessage("無效的尺寸，請輸入大於0且建議為32倍數的數字", 3000)
        except ValueError:
            win.statusBar().showMessage("請輸入有效的數字", 3000)

    def on_lock_key_changed(self, text):
        win = self.main_window
        win.lock_key = text
        win.auto_lock_mode = not win.lock_key  # 若 lock_key 為空，則啟用自動鎖定
        Config.content["lockKey"] = win.lock_key
        logger.info(f"鎖定按鍵更改為: {win.lock_key}")
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.auto_lock_mode = win.auto_lock_mode
            logger.info(f"自動鎖定模式已更新為: {win.auto_lock_mode}")

    def on_side_key_lock_enabled_changed(self, state):
        win = self.main_window
        win.side_key_lock_enabled = bool(state)
        Config.content["side_key_lock_enabled"] = win.side_key_lock_enabled
        status = '啟用' if win.side_key_lock_enabled else '禁用'
        logger.info(f"側鍵鎖定功能 {status}")
        win.statusBar().showMessage(f"側鍵鎖定功能 {status}", 3000)

    def on_side_lock_key_changed(self, text):
        win = self.main_window
        win.side_lock_key = text
        Config.content["side_lock_key"] = win.side_lock_key
        logger.info(f"側鍵鎖定按鍵更改為: {win.side_lock_key}")

    def on_prediction_checkbox_state_changed(self, state):
        win = self.main_window
        win.prediction_enabled = bool(state)
        Config.content["prediction"] = win.prediction_enabled
        status = '啟用' if win.prediction_enabled else '禁用'
        logger.info(f"預測瞄準功能 {status}")
        win.statusBar().showMessage(f"預測瞄準功能 {status}", 3000)
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()
            win.new_aim_logic.reset_state()


    def on_prediction_time_slider_value_changed(self, value):
        win = self.main_window
        win.prediction_time = value / 10.0
        win.prediction_time_value_label.setText(f"{win.prediction_time:.1f}")
        Config.content["prediction_time"] = win.prediction_time
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_auto_scale_aim_range_changed(self, state):
        win = self.main_window
        win.auto_scale_aim_range = bool(state)
        Config.content["auto_scale_aim_range"] = win.auto_scale_aim_range
        win.auto_scale_factor_slider.setEnabled(win.auto_scale_aim_range)
        win.auto_scale_factor_label.setEnabled(win.auto_scale_aim_range)
        status = '啟用' if win.auto_scale_aim_range else '禁用'
        logger.info(f"自動縮放瞄準範圍功能 {status}")
        win.statusBar().showMessage(f"自動縮放瞄準範圍功能 {status}", 3000)
        if not win.auto_scale_aim_range:
            win.aim_range = win.original_aim_range
            win.aim_range_slider.setValue(win.aim_range)

    def on_auto_scale_factor_changed(self, value):
        win = self.main_window
        win.auto_scale_factor = value / 10.0
        win.auto_scale_factor_value_label.setText(f"{win.auto_scale_factor:.1f}")
        Config.content["auto_scale_factor"] = win.auto_scale_factor

    def on_mouse_thread_changed(self, state):
        win = self.main_window
        win.mouse_threading = bool(state)
        Config.content["mouse_threading"] = win.mouse_threading
        status = '啟用' if win.mouse_threading else '禁用'
        logger.info(f"滑鼠多執行緒功能 {status}")
        win.statusBar().showMessage(f"滑鼠多執行緒功能 {status}", 3000)

    def on_mouse_move_mode_changed(self, text):
        win = self.main_window
        if text == "按鍵魔盒-鍵鼠轉接器":
            control.init_km_driver()
            if not control.is_km_driver_ready():
                QMessageBox.warning(win, "設備未就緒", "未檢測到 '按鍵魔盒-鍵鼠轉接器' 設備，已自動切換至 'win32' 滑鼠模式。")
                win.mouse_move_mode = "win32"
                win.mouse_move_mode_combobox.setCurrentText("win32")
            else:
                win.mouse_move_mode = text
                logger.info(f"滑鼠移動模式更改為: {win.mouse_move_mode}")
        elif text == "hid007":
            control.init_hid_driver()
            if not control.is_hid_driver_ready():
                QMessageBox.warning(win, "設備未就緒", "未檢測到 'hid007' 設備，已自動切換至 'win32' 滑鼠模式。")
                win.mouse_move_mode = "win32"
                win.mouse_move_mode_combobox.setCurrentText("win32")
            else:
                win.mouse_move_mode = text
                logger.info(f"滑鼠移動模式更改為: {win.mouse_move_mode}")
        else:
            win.mouse_move_mode = text
            logger.info(f"滑鼠移動模式更改為: {win.mouse_move_mode}")
        
        Config.content["mouse_move_mode"] = win.mouse_move_mode

    def toggle_aim_overlay(self):
        win = self.main_window
        if win.toggle_aim_overlay_button.isChecked():
            if not win.aim_overlay_window:
                win.aim_overlay_window = AimOverlayWindow(
                    win.aim_range,
                    color=win.overlay_color,
                    lock_color=win.overlay_lock_color
                )
            self.update_color_previews()
            win.aim_overlay_window.show()
            win.aim_overlay_window.center_on_screen()
            win.aim_overlay_enabled = True
            win.toggle_aim_overlay_button.setText("關閉範圍")
            logger.info("瞄準範圍已開啟。")
        else:
            if win.aim_overlay_window:
                win.aim_overlay_window.hide()
            win.aim_overlay_enabled = False
            win.toggle_aim_overlay_button.setText("開啟範圍")
            logger.info("瞄準範圍已關閉。")

    def on_recoil_control_changed(self, state):
        win = self.main_window
        win.recoil_control_enabled = bool(state)
        Config.content["recoil_control_enabled"] = win.recoil_control_enabled
        win.recoil_x_strength_slider.setEnabled(win.recoil_control_enabled)
        win.recoil_y_strength_slider.setEnabled(win.recoil_control_enabled)
        win.recoil_delay_slider.setEnabled(win.recoil_control_enabled)
        win.recoil_trigger_keys_input.setEnabled(win.recoil_control_enabled)
        win.update_recoil_control()
        status = '啟用' if win.recoil_control_enabled else '禁用'
        logger.info(f"壓槍功能 {status}")
        win.statusBar().showMessage(f"壓槍功能 {status}", 3000)

    def on_recoil_x_strength_changed(self, value):
        win = self.main_window
        win.recoil_x_strength = value
        win.recoil_x_strength_value_label.setText(str(win.recoil_x_strength))
        Config.content["recoil_x_strength"] = win.recoil_x_strength
        win.update_recoil_control()

    def on_recoil_y_strength_changed(self, value):
        win = self.main_window
        win.recoil_y_strength = value
        win.recoil_y_strength_value_label.setText(str(win.recoil_y_strength))
        Config.content["recoil_y_strength"] = win.recoil_y_strength
        win.update_recoil_control()

    def on_recoil_delay_changed(self, value):
        win = self.main_window
        win.recoil_delay = value
        win.recoil_delay_value_label.setText(str(win.recoil_delay))
        Config.content["recoil_delay"] = win.recoil_delay
        win.update_recoil_control()

    def on_recoil_trigger_keys_changed(self, text):
        win = self.main_window
        win.recoil_trigger_keys = text
        Config.content["recoil_trigger_keys"] = win.recoil_trigger_keys
        win.update_recoil_control()

    def export_settings(self):
        win = self.main_window
        
        # 使用 settings_manager 儲存目前 UI 的設定
        win.settings_manager.save_settings_from_ui()
        
        settings_to_export = Config.content.copy()
        
        filePath, _ = QFileDialog.getSaveFileName(win, "導出設定", "", "JSON Files (*.json);;All Files (*)")
        
        if filePath:
            try:
                with open(filePath, 'w', encoding='utf-8') as f:
                    json.dump(settings_to_export, f, ensure_ascii=False, indent=4)
                
                config_filename = os.path.basename(filePath)
                Config.current_config_file = config_filename
                
                if config_filename == "settings.json":
                    win.config_file_display.setText("預設設定")
                else:
                    config_name = os.path.splitext(config_filename)[0]
                    win.config_file_display.setText(f"{config_name} 配置")
                
                win.current_config_path = filePath
                win.statusBar().showMessage(f"設定已成功導出至 {os.path.basename(filePath)}", 5000)
                logger.info(f"設定已導出至: {filePath}，當前配置文件: {Config.current_config_file}")
            except Exception as e:
                QMessageBox.critical(win, "導出失敗", f"無法儲存設定檔案：\n{e}")
                logger.error(f"導出設定失敗: {e}")

    def import_settings(self):
        win = self.main_window
        filePath, _ = QFileDialog.getOpenFileName(win, "導入設定", "", "JSON Files (*.json);;All Files (*)")
        
        if filePath:
            try:
                with open(filePath, 'r', encoding='utf-8') as f:
                    new_settings = json.load(f)
                
                # 確保必要的設定項存在
                if 'capture_fps' not in new_settings:
                    new_settings['capture_fps'] = Config.default.get('capture_fps', 240)
                    logger.info("導入的配置缺少 capture_fps 設定，已使用預設值")
                
                # 確保內建模型相關設定項存在
                if 'model_type' not in new_settings:
                    new_settings['model_type'] = Config.default.get('model_type', 'custom')
                    logger.info("導入的配置缺少 model_type 設定，已使用預設值")
                
                if 'selected_builtin_model' not in new_settings:
                    new_settings['selected_builtin_model'] = Config.default.get('selected_builtin_model', '')
                    logger.info("導入的配置缺少 selected_builtin_model 設定，已使用預設值")
                
                if 'custom_model_file' not in new_settings:
                    new_settings['custom_model_file'] = Config.default.get('custom_model_file', 'yolov8n.pt')
                    logger.info("導入的配置缺少 custom_model_file 設定，已使用預設值")
                
                if 'yolo_version' not in new_settings:
                    new_settings['yolo_version'] = Config.default.get('yolo_version', 'auto')
                    logger.info("導入的配置缺少 yolo_version 設定，已使用預設值")
                
                Config.content.update(new_settings)
                
                config_filename = os.path.basename(filePath)
                Config.current_config_file = config_filename
                
                Config.save_to_current_config()
                
                # 重新載入內建模型列表（以防模型檔案變更）
                win.builtin_models = win.load_builtin_models()
                
                # 先更新設定中的模型類型和相關設定
                imported_model_type = new_settings.get('model_type', 'custom')
                win.model_type = imported_model_type
                win.selected_builtin_model = new_settings.get('selected_builtin_model', '')
                
                # 使用 settings_manager 載入設定並套用至 UI
                win.settings_manager.load_settings_to_ui()
                
                if config_filename == "settings.json":
                    win.config_file_display.setText("預設設定")
                else:
                    config_name = os.path.splitext(config_filename)[0]
                    win.config_file_display.setText(f"{config_name} 配置")
                
                win.current_config_path = filePath
                
                # 根據匯入的模型類型顯示對應資訊
                model_type = new_settings.get('model_type', 'custom')
                yolo_version = new_settings.get('yolo_version', 'auto')
                
                # 轉換 YOLO 版本顯示格式
                version_display = {"none": "未選擇", "v5": "YOLOv5", "v8": "YOLOv8", "auto": "自動"}.get(yolo_version, yolo_version)
                
                if model_type == 'builtin':
                    selected_model = new_settings.get('selected_builtin_model', '')
                    if selected_model and selected_model in win.builtin_models:
                        win.statusBar().showMessage(f"設定已導入，使用內建模型: {selected_model} ({version_display})", 5000)
                    else:
                        win.statusBar().showMessage(f"設定已導入，但內建模型 '{selected_model}' 不存在 ({version_display})", 5000)
                else:
                    model_file = new_settings.get('model_file', '')
                    win.statusBar().showMessage(f"設定已導入，使用自訂模型: {os.path.basename(model_file)} ({version_display})", 5000)
                
                logger.info(f"設定已從 {filePath} 導入，當前配置文件: {Config.current_config_file}")
                logger.info(f"模型類型: {model_type}, YOLO版本: {yolo_version}, 模型: {new_settings.get('selected_builtin_model' if model_type == 'builtin' else 'model_file', 'N/A')}")

            except json.JSONDecodeError:
                QMessageBox.critical(win, "導入失敗", "檔案格式錯誤，無法解析 JSON。")
                logger.error(f"導入設定失敗: JSON 格式錯誤於檔案 {filePath}")
            except Exception as e:
                QMessageBox.critical(win, "導入失敗", f"讀取或應用設定時發生錯誤：\n{e}")
                logger.error(f"導入設定失敗: {e}")

    def save_current_config(self):
        win = self.main_window
        try:
            win.save_settings()
        except Exception as e:
            QMessageBox.critical(win, "保存失敗", f"無法保存設定檔案：\n{e}")
            logger.error(f"保存當前設定失敗: {e}")
    

    def update_color_previews(self):
        """更新顏色設定UI的顯示"""
        win = self.main_window
        if hasattr(win, 'overlay_color_preview'):
            r, g, b, a = win.overlay_color
            win.overlay_color_preview.setStyleSheet(f"background-color: rgba({r}, {g}, {b}, {a/255.0}); border: 1px solid gray;")
        if hasattr(win, 'lock_color_preview'):
            r, g, b, a = win.overlay_lock_color
            win.lock_color_preview.setStyleSheet(f"background-color: rgba({r}, {g}, {b}, {a/255.0}); border: 1px solid gray;")

    def choose_overlay_color(self):
        """開啟顏色選擇對話框來設定一般顏色"""
        win = self.main_window
        r, g, b, a = win.overlay_color
        initial_color = QColor(r, g, b, a)
        color = QColorDialog.getColor(initial_color, win, "選擇一般範圍顏色", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        
        if color.isValid():
            win.overlay_color = (color.red(), color.green(), color.blue(), color.alpha())
            self.update_color_previews()
            if win.aim_overlay_window:
                win.aim_overlay_window.set_colors(win.overlay_color, win.overlay_lock_color)
                win.aim_overlay_window.update()

    def choose_lock_color(self):
        """開啟顏色選擇對話框來設定鎖定顏色"""
        win = self.main_window
        r, g, b, a = win.overlay_lock_color
        initial_color = QColor(r, g, b, a)
        color = QColorDialog.getColor(initial_color, win, "選擇鎖定目標顏色", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        
        if color.isValid():
            win.overlay_lock_color = (color.red(), color.green(), color.blue(), color.alpha())
            self.update_color_previews()
            if win.aim_overlay_window:
                win.aim_overlay_window.set_colors(win.overlay_color, win.overlay_lock_color)
                win.aim_overlay_window.update()

    def update_and_apply_colors(self):
        """更新UI、設定檔和浮水印視窗"""
        win = self.main_window
        self.update_color_previews()
        Config.content["overlay_color_r"], Config.content["overlay_color_g"], Config.content["overlay_color_b"], Config.content["overlay_color_a"] = win.overlay_color
        Config.content["overlay_lock_color_r"], Config.content["overlay_lock_color_g"], Config.content["overlay_lock_color_b"], Config.content["overlay_lock_color_a"] = win.overlay_lock_color
        if win.aim_overlay_window:
            win.aim_overlay_window.set_colors(win.overlay_color, win.overlay_lock_color)
            win.aim_overlay_window.update()

    def on_show_fps_overlay_changed(self, state):
        win = self.main_window
        win.toggle_fps_overlay(bool(state))
        Config.content["show_fps_overlay"] = bool(state)
        status = '啟用' if bool(state) else '禁用'
        logger.info(f"FPS 浮水印 {status}")
        win.statusBar().showMessage(f"FPS 浮水印 {status}", 3000)

    def on_show_detection_boxes_changed(self, state):
        win = self.main_window
        win.show_detection_boxes = bool(state)
        Config.content["show_detection_boxes"] = win.show_detection_boxes
        if win.show_detection_boxes:
            if win.draw_screen_window:
                win.draw_screen_window.show_overlay()
            logger.info("偵測方塊已啟用。")
            win.statusBar().showMessage("偵測方塊已啟用", 3000)
        else:
            if win.draw_screen_window:
                win.draw_screen_window.hide_overlay()
            logger.info("偵測方塊已停用。")
            win.statusBar().showMessage("偵測方塊已停用", 3000)

    def on_capture_fps_changed(self, value):
        win = self.main_window
        win.capture_fps = value
        win.capture_fps_value_label.setText(str(win.capture_fps))
        Config.content["capture_fps"] = win.capture_fps
        
        if hasattr(win, 'capture_manager') and win.capture_manager:
            win.capture_manager.target_fps = win.capture_fps
            logger.info(f"截圖速率已更新為: {win.capture_fps} FPS")
            
        if win.capture_fps >= 200:
            performance_tip = "極致性能 - 高CPU使用率"
        elif win.capture_fps >= 120:
            performance_tip = "高性能 - 中等CPU使用率"
        elif win.capture_fps >= 60:
            performance_tip = "平衡性能 - 低CPU使用率"
        else:
            performance_tip = "節能模式 - 最低CPU使用率"
            
        win.statusBar().showMessage(f"截圖速率: {win.capture_fps} FPS ({performance_tip}) - 已保存", 5000)

    def on_iou_threshold_changed(self, value):
        """處理 IOU 閾值滑桿變更事件"""
        win = self.main_window
        win.iou_threshold = value / 100.0
        win.iou_threshold_value_label.setText(f"{win.iou_threshold:.2f}")
        Config.content["iou_threshold"] = win.iou_threshold
        
        if hasattr(win, 'nms_processor') and win.nms_processor:
            win.adaptive_nms_threshold = win.iou_threshold
            logger.info(f"NMS IOU閾值已更新為: {win.iou_threshold:.2f}")
            
        if win.iou_threshold >= 0.7:
            iou_tip = "嚴格模式 - 保留更多重疊檢測"
        elif win.iou_threshold >= 0.5:
            iou_tip = "標準模式 - 平衡檢測精度"
        elif win.iou_threshold >= 0.3:
            iou_tip = "寬鬆模式 - 去除更多重疊檢測"
        else:
            iou_tip = "極寬鬆模式 - 最大程度去除重疊"
            
        win.statusBar().showMessage(f"IOU閾值: {win.iou_threshold:.2f} ({iou_tip}) - 已保存", 5000)
