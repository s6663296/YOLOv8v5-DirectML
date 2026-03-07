import json
import os
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QMenu, QWidgetAction, QCheckBox, QWidget, QHBoxLayout
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QColorDialog
from Module.aim_overlay import AimOverlayWindow
from Module.config import Config
from Module.inference_backend import (
    backend_from_ui_text,
    backend_to_ui_text,
)
from Module.logger import logger
import Module.control as control

class UIHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self._target_class_menu = None  # 目標類別下拉選單
        self._target_class_actions = {}  # class_id -> QAction 對應
        self._target_all_action = None   # ALL 全選項目

    def init_signals(self):
        """初始化 .ui 內各控制項的訊號連接。"""
        win = self.main_window
        win.choose_model_button.clicked.connect(self.choose_model)
        win.aimbot_checkbox.stateChanged.connect(self.on_aimbot_checkbox_state_changed)
        if hasattr(win, 'aimbot_hotkey_enabled_checkbox'):
            win.aimbot_hotkey_enabled_checkbox.stateChanged.connect(self.on_aimbot_hotkey_enabled_changed)
        if hasattr(win, 'aimbot_hotkey_input'):
            win.aimbot_hotkey_input.textChanged.connect(self.on_aimbot_hotkey_changed)
        if hasattr(win, 'aimbot_hotkey_detect_button'):
            win.aimbot_hotkey_detect_button.clicked.connect(self.on_aimbot_hotkey_detect_clicked)
        win.offset_mode_combobox.currentTextChanged.connect(self.on_offset_mode_changed)
        win.offset_centerx_slider.valueChanged.connect(self.on_offset_centerx_slider_value_changed)
        win.offset_centery_slider.valueChanged.connect(self.on_offset_centery_slider_value_changed)
        win.aim_width_slider.valueChanged.connect(self.on_aim_width_changed)
        win.aim_height_slider.valueChanged.connect(self.on_aim_height_changed)
        win.aim_speed_x_slider.valueChanged.connect(self.on_aim_speed_x_slider_value_changed)
        win.aim_speed_y_slider.valueChanged.connect(self.on_aim_speed_y_slider_value_changed)

        win.yolo_confidence_slider.valueChanged.connect(self.on_yolo_confidence_slider_value_changed)
        win.iou_threshold_slider.valueChanged.connect(self.on_iou_threshold_slider_value_changed)
        win.model_size_input.textChanged.connect(self.on_model_size_changed)
        if hasattr(win, 'target_class_button'):
            win.target_class_button.clicked.connect(self._show_target_class_menu)
        win.lock_key_input.textChanged.connect(self.on_lock_key_changed)
        # ??菜葫??靽∟???
        if hasattr(win, 'lock_key_detect_button'):
            win.lock_key_detect_button.clicked.connect(self.on_lock_key_detect_clicked)
        win.toggle_yolo_button.clicked.connect(self.main_window.toggle_yolo)
        win.toggle_aim_overlay_button.clicked.connect(self.toggle_aim_overlay)
        win.side_key_lock_enabled_checkbox.stateChanged.connect(self.on_side_key_lock_enabled_changed)
        win.side_lock_key_input.textChanged.connect(self.on_side_lock_key_changed)
        if hasattr(win, 'side_lock_key_detect_button'):
            win.side_lock_key_detect_button.clicked.connect(self.on_side_lock_key_detect_clicked)
        win.exclude_side_key_offset_checkbox.stateChanged.connect(self.on_exclude_side_key_offset_changed)
        win.prediction_checkbox.stateChanged.connect(self.on_prediction_checkbox_state_changed)
        win.auto_scale_aim_range_checkbox.stateChanged.connect(self.on_auto_scale_aim_range_changed)
        win.auto_scale_factor_slider.valueChanged.connect(self.on_auto_scale_factor_changed)
        win.prediction_time_slider.valueChanged.connect(self.on_prediction_time_slider_value_changed)
        win.mouse_move_mode_combobox.currentTextChanged.connect(self.on_mouse_move_mode_changed)
        if hasattr(win, 'mouse_thread_checkbox'):
            win.mouse_thread_checkbox.stateChanged.connect(self.on_mouse_thread_changed)
        else:
            logger.warning("UI missing widget: mouse_thread_checkbox")
        win.visual_offset_changed.connect(win.update_visual_offset_ui)
        # 敺漣??嗡縑??
        win.recoil_control_checkbox.stateChanged.connect(self.on_recoil_control_changed)
        win.recoil_x_strength_slider.valueChanged.connect(self.on_recoil_x_strength_changed)
        win.recoil_y_strength_slider.valueChanged.connect(self.on_recoil_y_strength_changed)
        win.recoil_delay_slider.valueChanged.connect(self.on_recoil_delay_changed)
        win.recoil_trigger_keys_input.textChanged.connect(self.on_recoil_trigger_keys_changed)
        if hasattr(win, 'recoil_trigger_keys_detect_button'):
            win.recoil_trigger_keys_detect_button.clicked.connect(self.on_recoil_trigger_keys_detect_clicked)

        # 憿閮剖?靽∟?
        if hasattr(win, 'choose_overlay_color_button'):
            win.choose_overlay_color_button.clicked.connect(self.choose_overlay_color)
        else:
            logger.warning("UI missing widget: choose_overlay_color_button")
        if hasattr(win, 'choose_lock_color_button'):
            win.choose_lock_color_button.clicked.connect(self.choose_lock_color)
        else:
            logger.warning("UI missing widget: choose_lock_color_button")


        if hasattr(win, 'show_fps_overlay_checkbox'):
           win.show_fps_overlay_checkbox.stateChanged.connect(self.on_show_fps_overlay_changed)
        else:
           logger.warning("UI missing widget: show_fps_overlay_checkbox")

        if hasattr(win, 'show_latency_overlay_checkbox'):
           win.show_latency_overlay_checkbox.stateChanged.connect(self.on_show_latency_overlay_changed)
        else:
           logger.warning("UI missing widget: show_latency_overlay_checkbox")

        if hasattr(win, 'show_detection_boxes_checkbox'):
           win.show_detection_boxes_checkbox.stateChanged.connect(self.on_show_detection_boxes_changed)
        else:
           logger.warning("UI missing widget: show_detection_boxes_checkbox")
           
        if hasattr(win, 'capture_fps_slider'):
           win.capture_fps_slider.valueChanged.connect(self.on_capture_fps_changed)
        else:
           logger.warning("UI missing widget: capture_fps_slider")
        
        if hasattr(win, 'capture_source_combobox'):
           win.capture_source_combobox.currentTextChanged.connect(self.on_capture_source_changed)
        else:
           logger.warning("UI missing widget: capture_source_combobox")

        if hasattr(win, 'capture_monitor_combobox'):
           win.capture_monitor_combobox.currentIndexChanged.connect(self.on_capture_monitor_changed)
        else:
           logger.warning("UI missing widget: capture_monitor_combobox")
        
        # OBS UDP 銝脫?閮剖?
        if hasattr(win, 'obs_ip_input'):
           win.obs_ip_input.textChanged.connect(self.on_obs_ip_changed)
        else:
           logger.warning("UI missing widget: obs_ip_input")
        
        if hasattr(win, 'obs_port_spinbox'):
           win.obs_port_spinbox.valueChanged.connect(self.on_obs_port_changed)
        else:
           logger.warning("UI missing widget: obs_port_spinbox")
        
        if hasattr(win, 'inference_backend_combobox'):
           win.inference_backend_combobox.currentTextChanged.connect(self.on_inference_backend_changed)
        else:
           logger.warning("UI missing widget: inference_backend_combobox")
           
        if hasattr(win, 'process_priority_combobox'):
           win.process_priority_combobox.currentTextChanged.connect(self.on_process_priority_changed)
        else:
           logger.warning("UI missing widget: process_priority_combobox")

        if hasattr(win, 'init_mouse_mode_button'):
           win.init_mouse_mode_button.clicked.connect(self.on_init_mouse_mode_clicked)
        else:
           logger.warning("UI missing widget: init_mouse_mode_button")

        # PID ?批?其縑??
        if hasattr(win, 'pid_checkbox'):
            win.pid_checkbox.stateChanged.connect(self.on_pid_enabled_changed)
        else:
            logger.warning("UI missing widget: pid_checkbox")
        
        if hasattr(win, 'pid_kp_slider'):
            win.pid_kp_slider.valueChanged.connect(self.on_pid_kp_changed)
        
        if hasattr(win, 'pid_ki_slider'):
            win.pid_ki_slider.valueChanged.connect(self.on_pid_ki_changed)
        if hasattr(win, 'pid_kd_slider'):
            win.pid_kd_slider.valueChanged.connect(self.on_pid_kd_changed)


    def on_init_mouse_mode_clicked(self):
       """Initialize mouse driver for current selected mode."""
       win = self.main_window
       
       current_mode = win.mouse_move_mode_combobox.currentText()
       mode_lower = current_mode.lower()
       
       if current_mode == "hid007":
           control.init_hid_driver()
           if control.is_hid_driver_ready():
               QMessageBox.information(win, "Init", "hid007 initialized successfully.")
           else:
               QMessageBox.warning(win, "Init Failed", "hid007 initialization failed.")
       elif current_mode == "Makcu":
           control.init_makcu_driver()
           if control.is_makcu_driver_ready():
               QMessageBox.information(win, "Init", "Makcu initialized successfully.")
           else:
               QMessageBox.warning(win, "Init Failed", "Makcu initialization failed.")
       elif "lite" in mode_lower:
           control.init_kmbox_lite_driver()
           if control.is_kmbox_lite_driver_ready():
               QMessageBox.information(win, "Init", "KMBox Lite initialized successfully.")
           else:
               QMessageBox.warning(win, "Init Failed", "KMBox Lite initialization failed.")
       else:
           control.init_km_driver()
           if control.is_km_driver_ready():
               QMessageBox.information(win, "Init", "KM driver initialized successfully.")
           else:
               QMessageBox.warning(win, "Init Failed", "KM driver initialization failed.")
       logger.info(f"Mouse mode init requested: {current_mode}")

    def choose_model(self):
        win = self.main_window
        file_dialog = QFileDialog(win)
        file_dialog.setNameFilter("YOLO Models (*.pt *.onnx *.engine)")
        file_dialog.setWindowTitle("選擇 YOLO 模型")
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            
            win.model_path_display.setText(selected_file)
            Config.content["model_file"] = selected_file

            if selected_file.lower().endswith(".engine") and hasattr(win, "inference_backend_combobox"):
                trt_text = backend_to_ui_text("trt")
                win.inference_backend_combobox.setCurrentText(trt_text)
                win.inference_backend = "trt"
                win.statusBar().showMessage("已切換為 TensorRT 後端，重新啟動 YOLO 後生效", 5000)
            else:
                win.statusBar().showMessage("模型已選擇，重新啟動 YOLO 後生效", 5000)
            
            logger.info(f"已手動選擇模型: {selected_file}")

    def on_aimbot_checkbox_state_changed(self, state):
        win = self.main_window
        win.aimbot_enabled = bool(state)
        status = "Enabled" if win.aimbot_enabled else "Disabled"
        logger.info(f"Aimbot {status}")
        win.statusBar().showMessage(f"Aimbot {status}. Remember to save settings.", 3000)



    def on_offset_mode_changed(self, text):
        win = self.main_window
        win.offset_mode = text
        
        is_pixel_mode = (win.offset_mode == "像素偏移")

        # ?寞?璅∪?閮剖?皛▼蝭???蝐?
        x_min, x_max = -100, 100
        y_min, y_max = -100, 100
        
        win.offset_centerx_slider.setMinimum(x_min)
        win.offset_centerx_slider.setMaximum(x_max)
        win.offset_centery_slider.setMinimum(y_min)
        win.offset_centery_slider.setMaximum(y_max)
        
        win.label.setText("X 偏移:" if is_pixel_mode else "X 比例偏移:")
        win.label_2.setText("Y 偏移:" if is_pixel_mode else "Y 比例偏移:")
        

        # ?寞??唳芋撘?敺???撅祆扯??交?潔蒂閮剖?皛▼
        win.offset_centerx_slider.blockSignals(True)
        win.offset_centery_slider.blockSignals(True)
        if is_pixel_mode:
            win.offset_centerx_slider.setValue(win.offset_centerx_pixel)
            win.offset_centery_slider.setValue(win.offset_centery_pixel)
            win.offset_centerx_value_label.setText(f"{win.offset_centerx_pixel}")
            win.offset_centery_value_label.setText(f"{win.offset_centery_pixel}")
        else:
            win.offset_centerx_slider.setValue(int(win.offset_centerx_ratio * 100))
            win.offset_centery_slider.setValue(int(win.offset_centery_ratio * 100))
            win.offset_centerx_value_label.setText(f"{win.offset_centerx_ratio:.2f}")
            win.offset_centery_value_label.setText(f"{win.offset_centery_ratio:.2f}")
        win.offset_centerx_slider.blockSignals(False)
        win.offset_centery_slider.blockSignals(False)

        logger.info(f"偏移模式已切換為: {win.offset_mode}")
        win.statusBar().showMessage(f"偏移模式: {win.offset_mode}", 3000)

        # ?單??湔???摩銝剔??
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_offset_centerx_slider_value_changed(self, value):
        win = self.main_window
        if win.offset_mode == "像素偏移":
            win.offset_centerx_pixel = value
            win.offset_centerx_value_label.setText(f"{value}")
        else:  # 瘥??宏
            win.offset_centerx_ratio = value / 100.0
            win.offset_centerx_value_label.setText(f"{win.offset_centerx_ratio:.2f}")
        
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_offset_centery_slider_value_changed(self, value):
        win = self.main_window
        if win.offset_mode == "像素偏移":
            win.offset_centery_pixel = value
            win.offset_centery_value_label.setText(f"{value}") # 憿舐內皛▼????
        else:  # 瘥??宏
            win.offset_centery_ratio = value / 100.0
            win.offset_centery_value_label.setText(f"{value / 100.0:.2f}")

        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_aim_width_changed(self, value):
        win = self.main_window
        # ?銝?頞?璅∪?憭批?
        if value > win.model_size:
            value = win.model_size
            win.aim_width_slider.blockSignals(True)
            win.aim_width_slider.setValue(value)
            win.aim_width_slider.blockSignals(False)
            
        win.aim_width = value
        win.original_aim_width = value
        win.aim_width_value_label.setText(str(value))
        
        if win.aim_overlay_window:
            win.aim_overlay_window.update_size(win.aim_width, win.aim_height)
            
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_aim_height_changed(self, value):
        win = self.main_window
        # ?銝?頞?璅∪?憭批?
        if value > win.model_size:
            value = win.model_size
            win.aim_height_slider.blockSignals(True)
            win.aim_height_slider.setValue(value)
            win.aim_height_slider.blockSignals(False)
            
        win.aim_height = value
        win.original_aim_height = value
        win.aim_height_value_label.setText(str(value))
        
        if win.aim_overlay_window:
            win.aim_overlay_window.update_size(win.aim_width, win.aim_height)

        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_aim_speed_x_slider_value_changed(self, value):
        win = self.main_window
        win.aim_speed_x = value / 10.0
        win.aim_speed_x_value_label.setText(f"{win.aim_speed_x:.1f}")
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_aim_speed_y_slider_value_changed(self, value):
        win = self.main_window
        win.aim_speed_y = value / 10.0
        win.aim_speed_y_value_label.setText(f"{win.aim_speed_y:.1f}")
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_detection_target_changed(self):
        """Update selected detection target classes."""
        win = self.main_window
        
        # ?園???◤?暸????ID
        selected_ids = []
        for class_id, checkbox in self._target_class_actions.items():
            if checkbox.isChecked():
                selected_ids.append(str(class_id))
        
        # ?湔 ALL ?暸???
        all_checked = len(selected_ids) == len(self._target_class_actions)
        if self._target_all_action:
            self._target_all_action.blockSignals(True)
            self._target_all_action.setChecked(all_checked)
            self._target_all_action.blockSignals(False)
        
        # 閮剖? target_class ??
        if all_checked or len(selected_ids) == 0:
            win.target_class = "ALL"
        else:
            win.target_class = ",".join(selected_ids)
        
        # ?湔????
        self._update_target_class_button_text()
        logger.info(f"目標類別已更新: {win.target_class}")
    
    def _on_target_all_toggled(self, checked):
        """Handle toggling of the ALL target option."""
        for checkbox in self._target_class_actions.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
        self.on_detection_target_changed()
    
    def _show_target_class_menu(self):
        """Show target class selection menu near the button."""
        win = self.main_window
        if not hasattr(win, 'target_class_button'):
            return
            
        if self._target_class_menu is None:
            # 憒??????交芋??憿舐內?內
            from PyQt6.QtWidgets import QToolTip
            btn = win.target_class_button
            QToolTip.showText(btn.mapToGlobal(QPoint(0, 0)), "Please start YOLO first.", btn)
            # ?蝙??statusBar
            win.statusBar().showMessage("Please start YOLO first.", 3000)
            logger.warning("Target class menu requested before YOLO initialization.")
            return

        btn = win.target_class_button
        # ?冽????孵???
        pos = btn.mapToGlobal(btn.rect().bottomLeft())
        self._target_class_menu.exec(pos)
    
    def setup_target_class_menu(self, model_names: dict):
        """依模型類別建立可多選的目標類別選單。"""
        win = self.main_window
        
        # 撱箇??詨
        self._target_class_menu = QMenu(win)
        
        # 頛?賢?嚗遣蝡???Checkbox ??QWidgetAction
        def create_checkable_action(text, callback, parent_menu):
            action = QWidgetAction(parent_menu)
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(8, 2, 8, 2)
            checkbox = QCheckBox(text)
            layout.addWidget(checkbox)
            widget.setLayout(layout)
            
            checkbox.toggled.connect(callback)
            action.setDefaultWidget(widget)
            parent_menu.addAction(action)
            return checkbox  # ? checkbox ?拐辣隞乩噶敺??批
        
        # ALL ?賊?
        self._target_all_action = create_checkable_action("ALL (全選)", self._on_target_all_toggled, self._target_class_menu)
        self._target_class_menu.addSeparator()
        
        # ???仿??
        self._target_class_actions = {}
        for class_id in sorted(model_names.keys()):
            class_name = model_names[class_id]
            text = f"{class_id}: {class_name}"
            # 瘜冽?嚗ambda ?刻艘?葉?閬?摰???
            checkbox = create_checkable_action(text, lambda checked, cid=class_id: self.on_detection_target_changed(), self._target_class_menu)
            self._target_class_actions[class_id] = checkbox
        
        # ?寞??嗅?閮剖??Ｗ儔?暸???
        self._restore_target_class_selection()
        self._update_target_class_button_text()
        
        logger.info(f"Target class menu initialized with {len(model_names)} classes.")
    
    def _restore_target_class_selection(self):
        """Restore target class selection from config."""
        win = self.main_window
        target_class = getattr(win, 'target_class', 'ALL')
        
        if target_class == "ALL":
            # ?券
            if self._target_all_action:
                self._target_all_action.setChecked(True)
            for checkbox in self._target_class_actions.values():
                checkbox.setChecked(True)
        else:
            # 閫????????ID
            try:
                selected_ids = set(int(x.strip()) for x in target_class.split(",") if x.strip())
            except ValueError:
                selected_ids = set()
            
            for class_id, checkbox in self._target_class_actions.items():
                checkbox.setChecked(class_id in selected_ids)
            
            if self._target_all_action:
                self._target_all_action.setChecked(len(selected_ids) == len(self._target_class_actions))
    
    def _update_target_class_button_text(self):
        """Update target class button text."""
        win = self.main_window
        if not hasattr(win, 'target_class_button'):
            return
        
        target_class = getattr(win, 'target_class', 'ALL')
        if target_class == "ALL":
            win.target_class_button.setText("目標類別: ALL")
        else:
            # 憿舐內撌脤憿?賊?
            count = len(target_class.split(","))
            total = len(self._target_class_actions)
            win.target_class_button.setText(f"目標類別: {count}/{total}")

    def on_yolo_confidence_slider_value_changed(self, value):
        win = self.main_window
        win.yolo_confidence = value / 100.0
        win.yolo_confidence_value_label.setText(f"{win.yolo_confidence:.2f}")
        logger.info(f"YOLO 置信度閾值: {win.yolo_confidence:.2f}")

    def on_iou_threshold_slider_value_changed(self, value):
        win = self.main_window
        win.iou_threshold = value / 100.0
        win.iou_threshold_value_label.setText(f"{win.iou_threshold:.2f}")
        logger.info(f"IOU 閾值: {win.iou_threshold:.2f}")

    def on_model_size_changed(self, text):
        win = self.main_window
        try:
            size = int(text)
            if size > 0 and size % 32 == 0:
                win.model_size = size
                logger.info(f"模型輸入尺寸已更新: {win.model_size}")
                win.capture_area = win.calculate_capture_area()
                win.update_video_label_size()
                win.statusBar().showMessage(f"模型尺寸已更新為 {win.model_size}", 3000)
                
                # ?湔??蝭??
                if hasattr(win, 'aim_width_slider'):
                    if win.aim_width > win.model_size:
                        win.aim_width_slider.setValue(win.model_size)
                if hasattr(win, 'aim_height_slider'):
                     if win.aim_height > win.model_size:
                        win.aim_height_slider.setValue(win.model_size)
            else:
                win.statusBar().showMessage("Model size must be positive and divisible by 32.", 3000)
        except ValueError:
            win.statusBar().showMessage("請輸入有效的數字", 3000)

    def on_lock_key_changed(self, text):
        win = self.main_window
        win.lock_key = text
        win.auto_lock_mode = not win.lock_key
        logger.info(f"主鎖定鍵已更新: {win.lock_key}")



        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.auto_lock_mode = win.auto_lock_mode
            logger.info(f"自動鎖定模式狀態: {win.auto_lock_mode}")

    def on_aimbot_hotkey_enabled_changed(self, state):
        win = self.main_window
        win.aimbot_hotkey_enabled = bool(state)
        if hasattr(win, 'update_aimbot_hotkey_listener'):
            win.update_aimbot_hotkey_listener()
        status = 'Enabled' if win.aimbot_hotkey_enabled else 'Disabled'
        logger.info(f"Aimbot toggle hotkey {status}")
        win.statusBar().showMessage(f"Aimbot toggle hotkey {status}", 3000)

    def on_aimbot_hotkey_changed(self, text):
        win = self.main_window
        win.aimbot_toggle_hotkey = text.strip().upper()
        if win.aimbot_hotkey_enabled and hasattr(win, 'update_aimbot_hotkey_listener'):
            win.update_aimbot_hotkey_listener()

    def on_aimbot_hotkey_detect_clicked(self):
        from Module.keyboard import capture_key_with_dialog
        win = self.main_window
        result = capture_key_with_dialog(win, allow_combo=False, title="Detect Aimbot Toggle Hotkey")
        if result:
            win.aimbot_hotkey_input.setText(result)
            logger.info(f"Detected aimbot toggle hotkey: {result}")

    def on_side_key_lock_enabled_changed(self, state):
        win = self.main_window
        win.side_key_lock_enabled = bool(state)
        status = '啟用' if win.side_key_lock_enabled else '停用'
        logger.info(f"側鍵鎖定已{status}")
        win.statusBar().showMessage(f"側鍵鎖定已{status}", 3000)

    def on_side_lock_key_changed(self, text):
        win = self.main_window
        win.side_lock_key = text
        logger.info(f"側鍵鎖定鍵已更新: {win.side_lock_key}")

    def on_lock_key_detect_clicked(self):
        """Detect main lock key."""
        from Module.keyboard import capture_key_with_dialog
        win = self.main_window
        result = capture_key_with_dialog(win, allow_combo=True, title="Detect Main Lock Key")
        if result:
            win.lock_key_input.setText(result)
            logger.info(f"Detected main lock key: {result}")

    def on_side_lock_key_detect_clicked(self):
        """Detect side lock key."""
        from Module.keyboard import capture_key_with_dialog
        win = self.main_window
        result = capture_key_with_dialog(win, allow_combo=True, title="Detect Side Lock Key")
        if result:
            win.side_lock_key_input.setText(result)
            logger.info(f"Detected side lock key: {result}")

    def on_recoil_trigger_keys_detect_clicked(self):
        """Detect recoil trigger keys."""
        from Module.keyboard import capture_key_with_dialog
        win = self.main_window
        result = capture_key_with_dialog(win, allow_combo=True, title="Detect Recoil Trigger Keys")
        if result:
            win.recoil_trigger_keys_input.setText(result)
            logger.info(f"Detected recoil trigger keys: {result}")

    def on_exclude_side_key_offset_changed(self, state):
        win = self.main_window
        win.exclude_side_key_offset = bool(state)
        status = '啟用' if win.exclude_side_key_offset else '停用'
        logger.info(f"排除側鍵偏移已{status}")
        win.statusBar().showMessage(f"排除側鍵偏移已{status}", 3000)
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_prediction_checkbox_state_changed(self, state):
        win = self.main_window
        win.prediction_enabled = bool(state)
        status = '啟用' if win.prediction_enabled else '停用'
        logger.info(f"預測功能已{status}")
        win.statusBar().showMessage(f"預測功能已{status}", 3000)
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()
            win.new_aim_logic.reset_state()


    def on_prediction_time_slider_value_changed(self, value):
        win = self.main_window
        win.prediction_time = value / 10.0
        win.prediction_time_value_label.setText(f"{win.prediction_time:.1f}")
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_auto_scale_aim_range_changed(self, state):
        win = self.main_window
        win.auto_scale_aim_range = bool(state)
        win.auto_scale_factor_slider.setEnabled(win.auto_scale_aim_range)
        win.auto_scale_factor_label.setEnabled(win.auto_scale_aim_range)
        status = '啟用' if win.auto_scale_aim_range else '停用'
        logger.info(f"自動縮放範圍已{status}")
        win.statusBar().showMessage(f"自動縮放範圍已{status}", 3000)
        if not win.auto_scale_aim_range:
            win.is_zoomed = False
            win.aim_dimensions_changed.emit(win.original_aim_width, win.original_aim_height)

    def on_auto_scale_factor_changed(self, value):
        win = self.main_window
        win.auto_scale_factor = value / 10.0
        win.auto_scale_factor_value_label.setText(f"{win.auto_scale_factor:.1f}")

    def on_mouse_thread_changed(self, state):
        win = self.main_window
        win.mouse_threading = bool(state)
        status = '啟用' if win.mouse_threading else '停用'
        logger.info(f"滑鼠執行緒已{status}")
        win.statusBar().showMessage(f"滑鼠執行緒已{status}", 3000)

    def on_mouse_move_mode_changed(self, text):
        win = self.main_window
        mode_lower = text.lower()

        if text == "hid007":
            control.init_hid_driver()
            if not control.is_hid_driver_ready():
                QMessageBox.warning(win, "Mode Error", "Failed to initialize hid007.")
                return
        elif text == "Makcu":
            control.init_makcu_driver()
            if not control.is_makcu_driver_ready():
                QMessageBox.warning(win, "Mode Error", "Failed to initialize Makcu.")
                return
        elif "lite" in mode_lower:
            control.init_kmbox_lite_driver()
            if not control.is_kmbox_lite_driver_ready():
                QMessageBox.warning(win, "Mode Error", "Failed to initialize KMBox Lite.")
                return
        elif "km" in mode_lower or "鍵" in text:
            control.init_km_driver()
            if not control.is_km_driver_ready():
                QMessageBox.warning(win, "Mode Error", "Failed to initialize KM driver.")
                return

        win.mouse_move_mode = text
        logger.info(f"Mouse move mode changed to: {win.mouse_move_mode}")
        

    def toggle_aim_overlay(self):
        win = self.main_window
        if win.toggle_aim_overlay_button.isChecked():
            if not win.aim_overlay_window:
                win.aim_overlay_window = AimOverlayWindow(
                    width=win.aim_width,
                    height=win.aim_height,
                    color=win.overlay_color,
                    lock_color=win.overlay_lock_color
                )
            self.update_color_previews()
            win.aim_overlay_window.show()
            win.aim_overlay_window.center_on_screen()
            win.aim_overlay_enabled = True
            win.toggle_aim_overlay_button.setText("關閉瞄準框")
            logger.info("Aim overlay enabled.")
        else:
            if win.aim_overlay_window:
                win.aim_overlay_window.hide()
            win.aim_overlay_enabled = False
            win.toggle_aim_overlay_button.setText("開啟瞄準框")
            logger.info("Aim overlay disabled.")

    def on_recoil_control_changed(self, state):
        win = self.main_window
        win.recoil_control_enabled = bool(state)
        win.recoil_x_strength_slider.setEnabled(win.recoil_control_enabled)
        win.recoil_y_strength_slider.setEnabled(win.recoil_control_enabled)
        win.recoil_delay_slider.setEnabled(win.recoil_control_enabled)
        win.recoil_trigger_keys_input.setEnabled(win.recoil_control_enabled)
        win.update_recoil_control()
        status = '啟用' if win.recoil_control_enabled else '停用'
        logger.info(f"後座力控制已{status}")
        win.statusBar().showMessage(f"後座力控制已{status}", 3000)

    def on_recoil_x_strength_changed(self, value):
        win = self.main_window
        win.recoil_x_strength = value
        win.recoil_x_strength_value_label.setText(str(win.recoil_x_strength))
        win.update_recoil_control()

    def on_recoil_y_strength_changed(self, value):
        win = self.main_window
        win.recoil_y_strength = value
        win.recoil_y_strength_value_label.setText(str(win.recoil_y_strength))
        win.update_recoil_control()

    def on_recoil_delay_changed(self, value):
        win = self.main_window
        win.recoil_delay = value
        win.recoil_delay_value_label.setText(str(win.recoil_delay))
        win.update_recoil_control()

    def on_recoil_trigger_keys_changed(self, text):
        win = self.main_window
        win.recoil_trigger_keys = text
        win.update_recoil_control()



    def export_settings(self):
        win = self.main_window
        
        # 雿輻 settings_manager ?脣??桀? UI ?身摰?
        win.settings_manager.save_settings_from_ui()
        
        settings_to_export = Config.content.copy()
        
        filePath, _ = QFileDialog.getSaveFileName(win, "匯出設定檔", "", "JSON Files (*.json);;All Files (*)")
        
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
                win.statusBar().showMessage(f"設定檔已匯出至 {os.path.basename(filePath)}", 5000)
                logger.info(f"設定檔已匯出至: {filePath}，目前啟用: {Config.current_config_file}")
            except Exception as e:
                QMessageBox.critical(win, "匯出失敗", f"無法匯出設定檔：\n{e}")
                logger.error(f"匯出設定檔失敗: {e}")

    def import_settings(self):
        win = self.main_window
        filePath, _ = QFileDialog.getOpenFileName(win, "Import Config", "", "JSON Files (*.json);;All Files (*)")

        if not filePath:
            return

        try:
            with open(filePath, 'r', encoding='utf-8') as f:
                new_settings = json.load(f)

            if 'capture_fps' not in new_settings:
                new_settings['capture_fps'] = Config.default.get('capture_fps', 240)
                logger.info("Imported config missing capture_fps. Default applied.")

            if 'yolo_version' not in new_settings:
                new_settings['yolo_version'] = Config.default.get('yolo_version', 'auto')
                logger.info("Imported config missing yolo_version. Default applied.")

            if 'capture_monitor_index' not in new_settings:
                new_settings['capture_monitor_index'] = Config.default.get('capture_monitor_index', 1)
                logger.info("Imported config missing capture_monitor_index. Default applied.")

            Config.content.update(new_settings)
            config_filename = os.path.basename(filePath)
            Config.current_config_file = config_filename
            Config.save_to_current_config()

            win.settings_manager.load_settings_to_ui()

            if config_filename == "settings.json":
                win.config_file_display.setText("Default Config")
            else:
                config_name = os.path.splitext(config_filename)[0]
                win.config_file_display.setText(f"{config_name} Config")

            win.current_config_path = filePath

            yolo_version = new_settings.get('yolo_version', 'auto')
            version_display = {
                "none": "None",
                "v5": "YOLOv5",
                "v8": "YOLOv8",
                "v11": "YOLOv11",
                "v12": "YOLOv12",
                "auto": "Auto",
            }.get(yolo_version, yolo_version)
            model_file = new_settings.get('model_file', '')
            win.statusBar().showMessage(
                f"Config imported: {os.path.basename(model_file)} ({version_display})",
                5000
            )

            logger.info(f"Config imported from {filePath}, active file: {Config.current_config_file}")
            logger.info(f"YOLO version: {yolo_version}, model: {model_file}")

        except json.JSONDecodeError:
            QMessageBox.critical(win, "Import Failed", "Invalid JSON format.")
            logger.error(f"Config import failed: invalid JSON: {filePath}")
        except Exception as e:
            QMessageBox.critical(win, "Import Failed", f"Failed to import config:\n{e}")
            logger.error(f"Config import failed: {e}")

    def save_current_config(self):
        win = self.main_window
        try:
            win.save_settings()
        except Exception as e:
            QMessageBox.critical(win, "儲存失敗", f"無法儲存設定檔：\n{e}")
            logger.error(f"儲存目前設定檔失敗: {e}")
    

    def update_color_previews(self):
        """Update color preview widgets."""
        win = self.main_window
        if hasattr(win, 'overlay_color_preview'):
            r, g, b, a = win.overlay_color
            win.overlay_color_preview.setStyleSheet(
                f"background-color: rgba({r}, {g}, {b}, {a/255.0}); border: 1px solid gray;"
            )
        if hasattr(win, 'lock_color_preview'):
            r, g, b, a = win.overlay_lock_color
            win.lock_color_preview.setStyleSheet(
                f"background-color: rgba({r}, {g}, {b}, {a/255.0}); border: 1px solid gray;"
            )

    def choose_overlay_color(self):
        """Choose overlay color."""
        win = self.main_window
        r, g, b, a = win.overlay_color
        initial_color = QColor(r, g, b, a)
        color = QColorDialog.getColor(
            initial_color,
            win,
            "Choose Overlay Color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel
        )

        if color.isValid():
            win.overlay_color = (color.red(), color.green(), color.blue(), color.alpha())
            self.update_color_previews()
            if win.aim_overlay_window:
                win.aim_overlay_window.set_colors(win.overlay_color, win.overlay_lock_color)
                win.aim_overlay_window.update()

    def choose_lock_color(self):
        """Choose lock color."""
        win = self.main_window
        r, g, b, a = win.overlay_lock_color
        initial_color = QColor(r, g, b, a)
        color = QColorDialog.getColor(
            initial_color,
            win,
            "Choose Lock Color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel
        )

        if color.isValid():
            win.overlay_lock_color = (color.red(), color.green(), color.blue(), color.alpha())
            self.update_color_previews()
            if win.aim_overlay_window:
                win.aim_overlay_window.set_colors(win.overlay_color, win.overlay_lock_color)
                win.aim_overlay_window.update()

    def on_show_fps_overlay_changed(self, state):
        win = self.main_window
        win.toggle_fps_overlay(bool(state))
        status = "Enabled" if bool(state) else "Disabled"
        logger.info(f"FPS overlay {status}")
        win.statusBar().showMessage(f"FPS overlay {status}", 3000)

    def on_show_latency_overlay_changed(self, state):
        win = self.main_window
        win.toggle_latency_overlay(bool(state))
        status = "Enabled" if bool(state) else "Disabled"
        logger.info(f"Latency overlay {status}")
        win.statusBar().showMessage(f"Latency overlay {status}", 3000)

    def on_show_detection_boxes_changed(self, state):
        win = self.main_window
        win.show_detection_boxes = bool(state)
        if win.show_detection_boxes:
            if win.draw_screen_window:
                win.draw_screen_window.show_overlay()
            if hasattr(win, '_push_latest_boxes_to_draw_queue'):
                win._push_latest_boxes_to_draw_queue([])
            logger.info("Detection boxes enabled")
            win.statusBar().showMessage("Detection boxes enabled", 3000)
        else:
            if hasattr(win, '_push_latest_boxes_to_draw_queue'):
                win._push_latest_boxes_to_draw_queue([])
            if win.draw_screen_window:
                win.draw_screen_window.update_boxes([])
                win.draw_screen_window.hide_overlay()
            logger.info("Detection boxes disabled")
            win.statusBar().showMessage("Detection boxes disabled", 3000)

    def on_capture_source_changed(self, text):
        """Handle capture source changed."""
        win = self.main_window
        if "OBS" in text:
            win.capture_source = "obs"
        elif "DXGI" in text:
            win.capture_source = "dxgi"
        else:
            win.capture_source = "mss"
        is_obs = (win.capture_source == "obs")

        widgets_to_toggle = [
            'obs_ip_label', 'obs_ip_input',
            'obs_port_label', 'obs_port_spinbox',
            'obs_info_label'
        ]
        for widget_name in widgets_to_toggle:
            if hasattr(win, widget_name):
                getattr(win, widget_name).setVisible(is_obs)

        if hasattr(win, 'capture_monitor_label'):
            win.capture_monitor_label.setEnabled(not is_obs)
        if hasattr(win, 'capture_monitor_combobox'):
            win.capture_monitor_combobox.setEnabled(not is_obs)

        logger.info(f"Capture source changed to: {win.capture_source}, OBS visible: {is_obs}")
        win.statusBar().showMessage(f"Capture source: {text}", 5000)

    def on_capture_monitor_changed(self, _index):
        """Handle capture monitor selection changed."""
        win = self.main_window

        if not hasattr(win, 'capture_monitor_combobox'):
            return

        selected_monitor = win.capture_monitor_combobox.currentData()
        if selected_monitor is None:
            selected_monitor = win.capture_monitor_combobox.currentIndex() + 1

        try:
            win.capture_monitor_index = max(1, int(selected_monitor))
        except (TypeError, ValueError):
            win.capture_monitor_index = 1

        if not win.yolo_enabled:
            win.capture_area = win.calculate_capture_area()

        monitor_info = win.get_monitor_by_index(win.capture_monitor_index) if hasattr(win, 'get_monitor_by_index') else None
        monitor_label = monitor_info['label'] if monitor_info else f"顯示器 {win.capture_monitor_index}"

        restart_hint = "（重啟 YOLO 後生效）" if win.yolo_enabled else ""
        win.statusBar().showMessage(f"截圖顯示器: {monitor_label}{restart_hint}", 5000)
        logger.info(f"Capture monitor changed to index={win.capture_monitor_index}, label={monitor_label}")

    def on_obs_ip_changed(self, text):
        """Handle OBS IP changed."""
        win = self.main_window
        win.obs_ip = text.strip()
        if win.obs_ip:
            logger.info(f"OBS IP set: {win.obs_ip}")
            win.statusBar().showMessage(f"OBS UDP IP: {win.obs_ip}", 5000)
        else:
            logger.info("OBS IP cleared")
            win.statusBar().showMessage("OBS UDP IP cleared", 5000)

    def on_obs_port_changed(self, value):
        """Handle OBS port changed."""
        win = self.main_window
        win.obs_port = value
        logger.info(f"OBS port set: {win.obs_port}")
        win.statusBar().showMessage(f"OBS UDP Port: {win.obs_port}", 5000)

    def on_inference_backend_changed(self, text):
        """Handle inference backend changed."""
        win = self.main_window
        win.inference_backend = backend_from_ui_text(text)

        model_path = win.model_path_display.text().strip() if hasattr(win, "model_path_display") else ""
        if win.inference_backend == "trt" and not model_path.lower().endswith(".engine"):
            win.statusBar().showMessage("TensorRT 模式需要 .engine 模型，請先參考 TensorRT_安裝指南.md 轉換模型", 6000)
        else:
            win.statusBar().showMessage(f"推理後端: {text}", 5000)

        logger.info(f"Inference backend changed to: {win.inference_backend}")

    def on_capture_fps_changed(self, value):
        win = self.main_window
        win.capture_fps = value
        win.capture_fps_value_label.setText(str(win.capture_fps))

        if hasattr(win, 'capture_manager') and win.capture_manager:
            win.capture_manager.target_fps = win.capture_fps
            logger.info(f"Capture FPS updated to: {win.capture_fps}")

            if win.yolo_enabled:
                if hasattr(win, 'get_ui_refresh_interval'):
                    update_interval = win.get_ui_refresh_interval(win.capture_fps)
                else:
                    update_interval = max(1, 1000 // win.capture_fps)
                win.video_timer.setInterval(update_interval)
                win.drawing_timer.setInterval(update_interval)
                logger.info(f"Preview refresh synced to {win.capture_fps} FPS ({update_interval}ms)")

        if win.capture_fps >= 200:
            performance_tip = "Very high"
        elif win.capture_fps >= 120:
            performance_tip = "High"
        elif win.capture_fps >= 60:
            performance_tip = "Balanced"
        else:
            performance_tip = "Power saving"

        win.statusBar().showMessage(f"Capture FPS: {win.capture_fps} ({performance_tip})", 5000)

    def on_process_priority_changed(self, text):
       """Handle process priority changed."""
       win = self.main_window
       win.process_priority = text
       logger.info(f"Process priority changed to: {text}")
       win.set_process_priority()
       win.statusBar().showMessage(f"Process priority set: {text}", 5000)

    # ====== PID handlers ======
    def on_pid_enabled_changed(self, state):
        """Handle PID enabled state changed."""
        win = self.main_window
        win.pid_enabled = bool(state)

        if hasattr(win, 'pid_kp_slider'):
            win.pid_kp_slider.setEnabled(win.pid_enabled)
        if hasattr(win, 'pid_ki_slider'):
            win.pid_ki_slider.setEnabled(win.pid_enabled)
        if hasattr(win, 'pid_kd_slider'):
            win.pid_kd_slider.setEnabled(win.pid_enabled)

        status = "Enabled" if win.pid_enabled else "Disabled"
        logger.info(f"PID {status}")
        win.statusBar().showMessage(f"PID {status}", 3000)

        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_pid_kp_changed(self, value):
        """Handle PID Kp changed."""
        win = self.main_window
        win.pid_kp = value
        kp_float = value / 100.0
        if hasattr(win, 'pid_kp_value_label'):
            win.pid_kp_value_label.setText(f"{kp_float:.2f}")
        logger.info(f"PID Kp updated: {kp_float:.2f}")
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_pid_ki_changed(self, value):
        """Handle PID Ki changed."""
        win = self.main_window
        win.pid_ki = value
        ki_float = value / 1000.0
        if hasattr(win, 'pid_ki_value_label'):
            win.pid_ki_value_label.setText(f"{ki_float:.3f}")
        logger.info(f"PID Ki updated: {ki_float:.3f}")
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()

    def on_pid_kd_changed(self, value):
        """Handle PID Kd changed."""
        win = self.main_window
        win.pid_kd = value
        kd_float = value / 1000.0
        if hasattr(win, 'pid_kd_value_label'):
            win.pid_kd_value_label.setText(f"{kd_float:.3f}")
        logger.info(f"PID Kd updated: {kd_float:.3f}")
        if hasattr(win, 'new_aim_logic') and win.new_aim_logic:
            win.new_aim_logic.update_parameters()
