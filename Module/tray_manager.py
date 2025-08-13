import sys
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QApplication
from PyQt6.QtGui import QIcon, QAction
from .logger import logger
import Module.control as control
from .utils import resource_path

class TrayManager:
    """處理系統匣圖示和應用程式生命週期事件。"""
    def __init__(self, main_window):
        """
        初始化系統匣管理器。

        :param main_window: 主視窗 (SudaneseboyApp) 的實例。
        """
        self.window = main_window
        self.tray_icon = None
        self.init_tray_icon()

    def init_tray_icon(self):
        """初始化系統匣圖示和選單。"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("系統不支援系統匣功能")
            return
            
        self.tray_icon = QSystemTrayIcon(self.window)
        self.tray_icon.setIcon(QIcon(resource_path("app.ico")))

        show_action = QAction("顯示", self.window)
        quit_action = QAction("退出", self.window)

        show_action.triggered.connect(self.show_window)
        quit_action.triggered.connect(self.quit_application)

        tray_menu = QMenu(self.window)
        tray_menu.addAction(show_action)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        
        self.tray_icon.activated.connect(self.tray_icon_activated)
        
        if self.tray_icon.isVisible():
            self.tray_icon.showMessage(
                "程式已啟動",
                "程式已在系統匣中運行，點擊圖示可開啟視窗。",
                QSystemTrayIcon.MessageIcon.Information,
                3000
            )
            logger.info("系統匣圖示初始化成功")
        else:
            logger.warning("系統匣圖示初始化失敗")

    def tray_icon_activated(self, reason):
        """處理系統匣圖示的點擊事件。"""
        try:
            if reason == QSystemTrayIcon.ActivationReason.Trigger:
                if self.window.isHidden():
                    self.show_window()
                else:
                    self.window.hide()
            elif reason == QSystemTrayIcon.ActivationReason.DoubleClick:
                if self.window.isHidden():
                    self.show_window()
        except Exception as e:
            logger.error(f"系統匣圖示點擊事件處理錯誤: {e}")
            try:
                self.show_window()
            except:
                pass

    def show_window(self):
        """顯示並啟用主視窗。"""
        self.window.showNormal()
        self.window.activateWindow()

    def quit_application(self):
        """完整地退出應用程式。"""
        self.window.exit_event.set()
        if hasattr(self.window, 'mouse_thread') and self.window.mouse_thread and self.window.mouse_thread.is_alive():
            self.window.mouse_thread.join(timeout=1)
            
        self.window.save_settings()
        
        # 關閉所有疊加層視窗
        if hasattr(self.window, 'aim_overlay_window') and self.window.aim_overlay_window:
           self.window.aim_overlay_window.close()
        if hasattr(self.window, 'fps_overlay_window') and self.window.fps_overlay_window:
           self.window.fps_overlay_window.close()
        if hasattr(self.window, 'draw_screen_window') and self.window.draw_screen_window:
           self.window.draw_screen_window.close()
        if hasattr(self.window, 'preview_window') and self.window.preview_window:
           self.window.preview_window.close()
      
        # 關閉硬體驅動
        if control.hiddll and control.is_hid_driver_ready():
            try:
                control.hiddll.close_hiddev()
                logger.info("hid007 設備已成功關閉。")
            except Exception as e:
                logger.error(f"關閉 hid007 設備時出錯: {e}")

        self.quit_application_internal()
        
        if self.tray_icon:
            self.tray_icon.hide()
            
        logger.info("應用程式關閉，設定已保存。")
        
        if hasattr(self.window, 'recoil_control'):
            self.window.recoil_control.stop()
            
        QApplication.instance().quit()

    def quit_application_internal(self):
        """清理應用程式內部資源。"""
        if hasattr(self.window, 'frame_manager'):
            stats = self.window.frame_manager.get_stats()
            logger.info(f"Frame manager stats: {stats}")
            self.window.frame_manager.cleanup()
        
        if hasattr(self.window, 'thread_pool_manager'):
            mouse_stats = self.window.thread_pool_manager.get_mouse_pool_stats()
            logger.info(f"Thread pool mouse stats: {mouse_stats}")
        
        if hasattr(self.window, 'nms_processor'):
            final_nms_stats = self.window.nms_processor.get_stats()
            logger.info(f"Final NMS stats: {final_nms_stats}")
        
        from Module.thread_pool_manager import shutdown_global_thread_pool
        shutdown_global_thread_pool()

    def handle_close_event(self, event):
        """處理主視窗的關閉事件 (X按鈕)。"""
        if self.tray_icon and self.tray_icon.isVisible():
            self.window.hide()
            try:
                self.tray_icon.showMessage(
                    "程式仍在背景執行",
                    "點擊系統匣圖示可還原視窗。",
                    QSystemTrayIcon.MessageIcon.Information,
                    3000
                )
            except Exception as e:
                logger.error(f"顯示系統匣通知時發生錯誤: {e}")
            
            logger.info("視窗已隱藏到系統匣")
            event.ignore()
        else:
            logger.info("系統匣不可用，正常關閉程式")
            self.quit_application_internal()
            event.accept()