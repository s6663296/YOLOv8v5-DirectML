import threading
import time
import win32api
import win32con
from Module.logger import logger
import Module.control as control
from Module.thread_pool_manager import get_thread_pool_manager

class RecoilControl:
    def __init__(self, app_instance):
        self.app = app_instance
        self.recoil_thread = None
        self.exit_event = threading.Event()
        self.enabled = False
        self.x_strength = 0
        self.y_strength = 5
        self.delay = 0.01
        self.trigger_keys = [] # To store VK_CODEs
        self.thread_pool_manager = get_thread_pool_manager()

    def set_config(self, enabled, x_strength, y_strength, delay, mouse_move_mode, trigger_keys_str):
        self.enabled = enabled
        self.x_strength = x_strength
        self.y_strength = y_strength
        self.delay = delay / 1000.0  # Convert ms to seconds
        self.mouse_move_mode = mouse_move_mode
        self._parse_trigger_keys(trigger_keys_str)
        
        if self.enabled:
            self.start()
        else:
            self.stop()

    def _parse_trigger_keys(self, keys_str):
        self.trigger_keys = []
        key_names = keys_str.strip().upper().split('+')
        for key_name in key_names:
            if key_name:
                try:
                    vk_code = getattr(win32con, key_name)
                    self.trigger_keys.append(vk_code)
                except AttributeError:
                    logger.warning(f"無效的按鍵名稱: {key_name}")
        logger.info(f"壓槍觸發按鍵設定為: {self.trigger_keys}")

    def _recoil_loop(self):
        logger.info("後座力控制執行緒已啟動。")
        while not self.exit_event.is_set():
            try:
                if self.trigger_keys:
                    all_keys_pressed = all(win32api.GetAsyncKeyState(key) & 0x8000 for key in self.trigger_keys)
                    if all_keys_pressed:
                        # 使用執行緒池進行後座力移動以避免阻塞
                        success = self.thread_pool_manager.submit_mouse_move(
                            self.mouse_move_mode, self.x_strength, self.y_strength, priority=2
                        )
                        if not success:
                            # 直接執行的備用方案
                            control.move(self.mouse_move_mode, self.x_strength, self.y_strength)
                
                # 等待指定的延遲
                time.sleep(self.delay)

            except Exception as e:
                logger.error(f"後座力控制迴圈出錯: {e}")
                # 出錯時等待更長時間以避免日誌垃圾郵件
                time.sleep(0.5)
        logger.info("後座力控制執行緒已停止。")

    def start(self):
        if self.recoil_thread is None or not self.recoil_thread.is_alive():
            self.exit_event.clear()
            self.recoil_thread = threading.Thread(target=self._recoil_loop, daemon=True)
            self.recoil_thread.start()
            logger.info("正在啟動後座力控制執行緒。")

    def stop(self):
        if self.recoil_thread and self.recoil_thread.is_alive():
            self.exit_event.set()
            self.recoil_thread = None
            logger.info("正在停止後座力控制執行緒。")