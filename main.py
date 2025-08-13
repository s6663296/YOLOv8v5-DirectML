import sys
import os
import ctypes


import cv2
import numpy as np
import pyautogui
import win32api
import win32con
import time
import threading
from math import sqrt
from collections import deque
from ultralytics import YOLO
import supervision as sv
from PyQt6 import QtWidgets, uic
import queue
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon, QAction
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QSlider, QComboBox, QCheckBox, QFileDialog, QMessageBox, QHBoxLayout, QLineEdit, QSystemTrayIcon, QMenu
import atexit

from Module.config import Config, Root
from Module.logger import logger
from Module.settings_manager import SettingsManager
from Module.tray_manager import TrayManager
import Module.control as control
from Module.aim_overlay import AimOverlayWindow
from Module.fps_overlay import FPSOverlay
from Module.draw_screen import DrawScreen
from Module.capture import ScreenCaptureManager
from Module.enhanced_inference import EnhancedInferenceManager
from Module.recoil_control import RecoilControl
from Module.frame_buffer_pool import FrameManager
from Module.thread_pool_manager import get_thread_pool_manager, shutdown_global_thread_pool
from Module.optimized_nms import optimized_nms_boxes, get_optimized_nms
from Module.new_aim_logic import NewAimLogic
from Module.ui_handler import UIHandler
from Module.preview_window import PreviewWindow
from Module.utils import resource_path

class main(QMainWindow):
    aim_range_changed = pyqtSignal(int)
    visual_offset_changed = pyqtSignal(float)
    fps_updated = pyqtSignal(int)
 
    def __init__(self):
        super().__init__()
        uic.loadUi(resource_path("main_window.ui"), self)
        self.setWindowIcon(QIcon(resource_path("app.ico")))
        
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.gridLayout_recoil.setColumnStretch(0, 0)
        self.gridLayout_recoil.setColumnStretch(1, 1)
        self.gridLayout_recoil.setColumnStretch(2, 0)
        
        self.settings_manager = SettingsManager(self)
        
        # 初始化屬性
        self.model = None
        self.yolo_enabled = False
        self.preview_enabled = True
        self.aim_overlay_enabled = False
        self.aim_overlay_window = None
        self.tracked_target = None
        self.last_target_time = 0
        self.current_config_path = None
        
        # 從 SettingsManager 載入初始設定
        self.settings_manager.load_initial_settings()

        self.recoil_control = RecoilControl(self)
        self.ui_handler = UIHandler(self)
        self.ui_handler.init_signals()
        
        self.close_button.clicked.connect(self.close)
        self.minimize_button.clicked.connect(self.showMinimized)

        self.fps_overlay_window = FPSOverlay()
        self.fps_updated.connect(self.fps_overlay_window.update_fps)

        self.draw_screen_window = DrawScreen()

        self.yolo_version_combobox.addItems(["請選擇YOLO版本", "YOLOv5", "YOLOv8"])
        self.yolo_version_combobox.currentTextChanged.connect(self.on_yolo_version_changed)

        # 內建模型相關UI元件已在 .ui 檔案中移除

        self.export_button.clicked.connect(self.ui_handler.export_settings)
        self.import_button.clicked.connect(self.ui_handler.import_settings)
        self.save_config_button.clicked.connect(self.ui_handler.save_current_config)

        self.preview_window = PreviewWindow()
        self.preview_window.hide()
        self.toggle_preview_button.clicked.connect(self.toggle_preview_window)
        
        self.load_settings()
        
        self.settings_manager.update_config_display()
        
        self.resize(470, 800)
        
        self.exit_event = threading.Event()
        self.new_frame_event = threading.Event()
        self.boxes_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.boxes_to_draw_queue = queue.Queue()
        
        self.capture_manager = None
        self.inference_manager = None
        self.processing_thread = None
        self.latest_processed_frame = None
        
        self.frame_manager = FrameManager()
        self.thread_pool_manager = get_thread_pool_manager()
        self.nms_processor = get_optimized_nms()
        
        self.nms_performance_history = deque(maxlen=100)
        self.adaptive_nms_threshold = self.iou_threshold
        
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_display)

        self.drawing_timer = QTimer(self)
        self.drawing_timer.timeout.connect(self.update_drawing_overlay)
        
        self.thread_pool_health_timer = QTimer(self)
        self.thread_pool_health_timer.timeout.connect(self.check_thread_pool_health)
        self.thread_pool_health_timer.start(30000)
        
        self.capture_area = self.calculate_capture_area()
 
        self.new_aim_logic = NewAimLogic(self)
        self.tracker = sv.ByteTrack()
        self.tracked_detections = None
 
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
 
        self.mouse_thread = threading.Thread(target=self.mouse_listener)
        self.mouse_thread.daemon = True
        self.mouse_thread.start()
        
        self.tray_manager = TrayManager(self)
        
        self.drag_position = None
 
    def load_builtin_models(self):
        """動態載入 Encrypted_Model 資料夾中的加密模型"""
        builtin_models = {}
        encrypted_model_dir = resource_path("Encrypted_Model")
        
        try:
            if os.path.exists(encrypted_model_dir):
                for filename in os.listdir(encrypted_model_dir):
                    if filename.lower().endswith('.eonnx'):
                        # 使用檔案名稱（不含副檔名）作為顯示名稱
                        model_name = os.path.splitext(filename)[0]
                        model_path = f"Encrypted_Model/{filename}"
                        builtin_models[model_name] = model_path
                        logger.info(f"發現內建模型: {model_name} -> {model_path}")
            else:
                logger.warning(f"內建模型資料夾不存在: {encrypted_model_dir}")
        except Exception as e:
            logger.error(f"載入內建模型時發生錯誤: {e}")
        
        if not builtin_models:
            logger.warning("未找到任何內建模型")
        else:
            logger.info(f"成功載入 {len(builtin_models)} 個內建模型")
        
        return builtin_models

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position is not None:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_position = None
        event.accept()

    def load_settings(self):
        """從 SettingsManager 載入設定並更新 UI"""
        self.settings_manager.load_settings_to_ui()

    def save_settings(self):
        """使用 SettingsManager 保存當前設定"""
        self.settings_manager.save_settings_from_ui()

    def on_yolo_version_changed(self, text):
        """處理YOLO版本變更事件"""
        version_map = {"請選擇YOLO版本": "none", "YOLOv5": "v5", "YOLOv8": "v8"}
        self.selected_yolo_version = version_map.get(text, "none")
        
        if self.selected_yolo_version in ["v5", "v8"]:
            self.statusBar().showMessage(f"YOLO 版本已設定為: {self.selected_yolo_version.upper()}，請保存設定以應用變更", 3000)
        else:
            self.statusBar().showMessage("請選擇一個有效的YOLO版本", 3000)

    def main_processing_loop(self):
        logger.info("Main processing loop started.")
        while not self.exit_event.is_set():
            self.frame_count += 1
            if time.time() - self.start_time >= 1.0:
                self.current_fps = self.frame_count
                self.frame_count = 0
                self.start_time = time.time()
                if self.show_fps_overlay_enabled:
                    self.fps_updated.emit(self.current_fps)
 
            try:
                if not self.new_frame_event.wait(timeout=1.0):
                    logger.debug("Timeout waiting for new frame.")
                    continue
                self.new_frame_event.clear()

                frame = self.capture_manager.get_frame()
                if frame is None:
                    continue
                
                processed_frame = self.frame_manager.get_processed_frame_buffer(frame)
                center_x_screen = self.capture_area["width"] // 2
                center_y_screen = self.capture_area["height"] // 2

                detection_results, inference_time = self.inference_manager.run_inference(frame)

                target_to_aim = None
                detections = sv.Detections.empty()

                if detection_results[0].size > 0:
                    boxes_xywh, scores, class_ids = detection_results
                    
                    model_input_size = self.inference_manager.input_shape[2]
                    capture_size = self.model_size
                    
                    if model_input_size > 0:
                        scale_factor = capture_size / model_input_size
                        if scale_factor != 1.0:
                            boxes_xywh *= scale_factor
                            logger.info(f"Unified Scaling: Applied scale factor {scale_factor:.4f} (model: {model_input_size}, capture: {capture_size})")

                    conf_mask = scores > self.yolo_confidence
                    
                    if self.target_class != "ALL":
                        try:
                            target_cls_id = int(self.target_class)
                            class_mask = class_ids == target_cls_id
                            final_mask = np.logical_and(conf_mask, class_mask)
                        except ValueError:
                            final_mask = conf_mask
                    else:
                        final_mask = conf_mask

                    if np.any(final_mask):
                        valid_boxes = boxes_xywh[final_mask]
                        valid_scores = scores[final_mask]
                        valid_class_ids = class_ids[final_mask]
                        
                        distances = np.sqrt((valid_boxes[:, 0] - center_x_screen)**2 + 
                                          (valid_boxes[:, 1] - center_y_screen)**2)
                        
                        in_range_mask = distances <= self.aim_range
                        
                        if np.any(in_range_mask):
                            in_range_distances = distances[in_range_mask]
                            closest_idx_in_range = np.argmin(in_range_distances)
                            
                            in_range_indices = np.where(in_range_mask)[0]
                            closest_idx = in_range_indices[closest_idx_in_range]
                            
                            closest_box = valid_boxes[closest_idx]
                            closest_score = valid_scores[closest_idx]
                            closest_class_id = valid_class_ids[closest_idx]
                            
                            closest_box_xyxy = np.array([[
                                closest_box[0] - closest_box[2] / 2,
                                closest_box[1] - closest_box[3] / 2,
                                closest_box[0] + closest_box[2] / 2,
                                closest_box[1] + closest_box[3] / 2
                            ]])
                            
                            detections = sv.Detections(
                                xyxy=closest_box_xyxy,
                                confidence=np.array([closest_score]),
                                class_id=np.array([closest_class_id])
                            )
                            
                            logger.debug(f"Tracking closest target: distance={distances[closest_idx]:.1f}px, confidence={closest_score:.2f}")
                        else:
                            detections = sv.Detections.empty()
                    else:
                        detections = sv.Detections.empty()

                self.tracked_detections = self.tracker.update_with_detections(detections)
                
                if len(self.tracked_detections) > 0:
                    box = self.tracked_detections.xyxy[0]
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    
                    track_id = self.tracked_detections.tracker_id[0] if hasattr(self.tracked_detections, 'tracker_id') and self.tracked_detections.tracker_id is not None else 0
                    
                    target_to_aim = {
                        'center_x': x_center,
                        'center_y': y_center,
                        'box_width': box[2] - box[0],
                        'box_height': box[3] - box[1],
                        'track_id': track_id
                    }

                self.new_aim_logic.process_data(target_to_aim)

                if self.show_detection_boxes and self.tracked_detections is not None and len(self.tracked_detections) > 0:
                    box = self.tracked_detections.xyxy[0]
                    screen_x1 = self.capture_area['left'] + box[0]
                    screen_y1 = self.capture_area['top'] + box[1]
                    screen_x2 = self.capture_area['left'] + box[2]
                    screen_y2 = self.capture_area['top'] + box[3]
                    boxes_to_draw = [(screen_x1, screen_y1, screen_x2, screen_y2)]
                    self.boxes_to_draw_queue.put(boxes_to_draw)
                elif self.show_detection_boxes:
                    self.boxes_to_draw_queue.put([])

                if self.preview_enabled and self.tracked_detections is not None and len(self.tracked_detections) > 0:
                    box = self.tracked_detections.xyxy[0]
                    confidence = self.tracked_detections.confidence[0] if self.tracked_detections.confidence is not None else 0.0
                    class_id = self.tracked_detections.class_id[0] if self.tracked_detections.class_id is not None else 0
                    track_id = self.tracked_detections.tracker_id[0] if hasattr(self.tracked_detections, 'tracker_id') and self.tracked_detections.tracker_id is not None else 0
                    
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    distance = sqrt((x_center - center_x_screen)**2 + (y_center - center_y_screen)**2)
                    
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if hasattr(self.inference_manager, 'model_names') and self.inference_manager.model_names:
                        class_name = self.inference_manager.model_names.get(class_id, f"Class_{class_id}")
                    else:
                        class_name = f"Class_{class_id}"
                    
                    yolo_ver = getattr(self.inference_manager, 'yolo_version', 'unknown')
                    label = f"{class_name}: {confidence:.3f} ID:{track_id} ({distance:.0f}px) [{yolo_ver}]"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    cv2.rectangle(processed_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    
                    cv2.putText(processed_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                if self.aim_overlay_enabled and self.aim_overlay_window:
                    self.aim_overlay_window.set_target_in_range(target_to_aim is not None)

                cv2.circle(processed_frame, (center_x_screen, center_y_screen), 5, (0, 0, 255), -1)
                cv2.circle(processed_frame, (center_x_screen, center_y_screen), self.aim_range, (255, 0, 0), 2)
                
                with self.frame_lock:
                    if self.latest_processed_frame is not None:
                        self.frame_manager.return_frame_buffer(self.latest_processed_frame)
                    self.latest_processed_frame = processed_frame

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
        
        if hasattr(self, 'nms_processor'):
            final_nms_stats = self.nms_processor.get_stats()
            logger.info(f"Final NMS stats: {final_nms_stats}")
            
        if hasattr(self, 'nms_performance_history') and self.nms_performance_history:
            final_app_nms_stats = self._analyze_nms_performance()
            logger.info(f"Final application NMS stats: {final_app_nms_stats}")

    def toggle_yolo(self):
        if self.yolo_enabled:
            self.yolo_enabled = False
            
            self.exit_event.set()
            self.video_timer.stop()
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1)

            if self.mouse_thread and self.mouse_thread.is_alive():
                 self.mouse_thread.join(timeout=1)
            
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
            if selected_version not in ['v5', 'v8']:
                QMessageBox.warning(self, "提示", "請先在下拉選單中選擇有效的YOLO版本 (v5或v8) 再啟動。")
                self.statusBar().showMessage("請選擇YOLO版本", 5000)
                return

            self.statusBar().showMessage("正在初始化...", 0)
            QApplication.processEvents()

            try:
                self.inference_manager = EnhancedInferenceManager(self, yolo_version=selected_version)
                if not self.inference_manager.initialize():
                    QMessageBox.critical(self, "錯誤", "ONNX 引擎初始化失敗，請檢查日誌。")
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
            if self.preview_enabled:
                self.preview_window.show()
            self.capture_area = self.calculate_capture_area()
            
            if hasattr(self, 'new_aim_logic') and self.new_aim_logic:
                self.new_aim_logic.update_parameters()

            self.exit_event.clear()
            
            self.capture_manager = ScreenCaptureManager(self.capture_area, self.exit_event, self.new_frame_event, self.capture_fps)
            self.capture_manager.start()

            self.processing_thread = threading.Thread(target=self.main_processing_loop, daemon=True)
            self.processing_thread.start()

            if not self.mouse_thread or not self.mouse_thread.is_alive():
                self.mouse_thread = threading.Thread(target=self.mouse_listener, daemon=True)
                self.mouse_thread.start()
                logger.info("Mouse listener thread restarted.")

            self.yolo_enabled = True
            
            self.video_timer.start(1000 // 60)
            self.drawing_timer.start(1000 // 60)
            self.toggle_yolo_button.setText("停止 YOLO")
            logger.info("YOLO pipeline started.")
            self.statusBar().showMessage("YOLO 已啟動", 3000)


    def toggle_preview_window(self):
        self.preview_enabled = not self.preview_enabled
        if self.preview_enabled and self.yolo_enabled:
            self.preview_window.show()
        else:
            self.preview_window.hide()
        self.toggle_preview_button.setText("關閉 YOLO 預覽" if self.preview_enabled else "開啟 YOLO 預覽")

    def toggle_always_on_top(self, checked):
        """此功能已棄用，預覽視窗永遠置頂，主視窗的置頂功能已移除"""
        pass

    def update_video_label_size(self):
        """根據模型大小動態調整預覽視窗中影像標籤的大小"""
        self.preview_window.video_label.setFixedSize(self.model_size, self.model_size)
        self.preview_window.resize(self.model_size, self.model_size)


    def update_aim_range_ui(self, new_range):
        self.aim_range = new_range
        self.aim_range_slider.blockSignals(True)
        self.aim_range_slider.setValue(self.aim_range)
        self.aim_range_slider.blockSignals(False)
        if self.aim_overlay_window:
            self.aim_overlay_window.set_aim_range(self.aim_range)
  
    def update_visual_offset_ui(self, new_offset_x):
        self.offset_centerx_slider.blockSignals(True)
        self.offset_centerx_slider.setValue(int(new_offset_x * 100))
        self.offset_centerx_slider.blockSignals(False)
        self.offset_centerx_value_label.setText(f"{new_offset_x:.2f}")

    def update_drawing_overlay(self):
        if not self.show_detection_boxes or not self.draw_screen_window:
            return
        
        try:
            boxes = self.boxes_to_draw_queue.get_nowait()
            self.draw_screen_window.update_boxes(boxes)
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
                    
                    if hasattr(self, 'nms_processor'):
                        nms_stats = self.nms_processor.get_stats()
                        logger.info(f"NMS performance stats: {nms_stats}")
                        self.nms_processor.reset_stats()
                        
                        if hasattr(self, 'nms_performance_history') and self.nms_performance_history:
                            app_nms_stats = self._analyze_nms_performance()
                            logger.info(f"Application NMS stats: {app_nms_stats}")
                            self.nms_performance_history.clear()
                    
                    
        except Exception as e:
            logger.error(f"Error in thread pool health check: {e}")
    
    def _analyze_nms_performance(self) -> dict:
        if not self.nms_performance_history:
            return {}
        
        history = list(self.nms_performance_history)
        
        input_boxes = [h['input_boxes'] for h in history]
        output_boxes = [h['output_boxes'] for h in history]
        nms_times = [h['nms_time'] for h in history]
        
        stats = {
            'total_calls': len(history),
            'avg_input_boxes': np.mean(input_boxes),
            'max_input_boxes': np.max(input_boxes),
            'avg_output_boxes': np.mean(output_boxes),
            'avg_nms_time_ms': np.mean(nms_times),
            'max_nms_time_ms': np.max(nms_times),
            'reduction_ratio': np.mean([o/i if i > 0 else 0 for i, o in zip(input_boxes, output_boxes)])
        }
        
        if stats['avg_nms_time_ms'] > 5.0:
            logger.warning(f"NMS performance degraded: avg time {stats['avg_nms_time_ms']:.2f}ms")
        
        if stats['max_nms_time_ms'] > 10.0:
            logger.warning(f"NMS spike detected: max time {stats['max_nms_time_ms']:.2f}ms")
        
        return stats


    def calculate_capture_area(self):
        screen_width, screen_height = pyautogui.size()
        capture_width, capture_height = self.model_size, self.model_size
        left = (screen_width - capture_width) // 2
        top = (screen_height - capture_height) // 2
        return {
            "top": top,
            "left": left,
            "width": capture_width,
            "height": capture_height
        }

    def update_display(self):
        if self.yolo_enabled and self.preview_enabled:
            with self.frame_lock:
                if self.latest_processed_frame is None:
                    return
                frame_to_display = self.frame_manager.get_display_frame_buffer(self.latest_processed_frame)

            fps_text = f"FPS: {self.current_fps}"
            cv2.putText(frame_to_display, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            h, w, ch = frame_to_display.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_to_display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.preview_window.video_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_window.video_label.setPixmap(scaled_pixmap)
            
            self.frame_manager.return_frame_buffer(frame_to_display)

    def mouse_listener(self):
        is_zoomed = False
        while not self.exit_event.is_set():
            try:
                if self.auto_scale_aim_range:
                    if win32api.GetAsyncKeyState(win32con.VK_RBUTTON) & 0x8000:
                        if not is_zoomed:
                            new_range = int(self.original_aim_range / (self.auto_scale_factor + 1))
                            self.aim_range_changed.emit(new_range)
                            is_zoomed = True
                            logger.debug(f"Zoom in signal emitted: aim_range set to {new_range}")
                    else:
                        if is_zoomed:
                            self.aim_range_changed.emit(self.original_aim_range)
                            is_zoomed = False
                            logger.debug(f"Zoom out signal emitted: aim_range restored to {self.original_aim_range}")
                else:
                    if is_zoomed:
                        self.aim_range_changed.emit(self.original_aim_range)
                        is_zoomed = False

                time.sleep(0.01)
            except Exception as e:
                logger.error(f"滑鼠監聽執行緒發生錯誤: {e}")
                time.sleep(1)

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


    def update_recoil_control(self):
        self.recoil_control.set_config(
            enabled=self.recoil_control_enabled,
            x_strength=self.recoil_x_strength,
            y_strength=self.recoil_y_strength,
            delay=self.recoil_delay,
            mouse_move_mode=self.mouse_move_mode,
            trigger_keys_str=self.recoil_trigger_keys
        )

    def closeEvent(self, event):
        """處理視窗關閉事件，將其委派給系統匣管理員"""
        if hasattr(self, 'tray_manager'):
            self.tray_manager.handle_close_event(event)
        else:
            # 若系統匣管理員不存在，則執行預設的關閉行為
            super().closeEvent(event)


    def showEvent(self, event):
        """處理視窗顯示事件"""
        super().showEvent(event)

    def hideEvent(self, event):
        """處理視窗隱藏事件"""
        super().hideEvent(event)


if __name__ == "__main__":
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    
    app.setQuitOnLastWindowClosed(False)
    
    window = main()
    window.show()
    sys.exit(app.exec())
