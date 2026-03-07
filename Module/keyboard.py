import ctypes
import tkinter as tk
import threading
import queue
from PyQt6 import uic
from PyQt6.QtWidgets import QDialog
from PyQt6.QtCore import Qt, QTimer

from Module.utils import resource_path


try:
    _user32 = ctypes.windll.user32
    _user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    _user32.GetAsyncKeyState.restype = ctypes.c_short
except AttributeError:
    _user32 = None


VK_ESCAPE = 0x1B
POLL_INTERVAL_MS = 16
CAPTURE_VK_CODES = tuple(range(0x01, 0xFF))


# 修飾鍵的 VK codes
MODIFIER_VK_CODES = {
    0x10,  # VK_SHIFT
    0x11,  # VK_CONTROL
    0x12,  # VK_MENU (Alt)
    0xA0,  # VK_LSHIFT
    0xA1,  # VK_RSHIFT
    0xA2,  # VK_LCONTROL
    0xA3,  # VK_RCONTROL
    0xA4,  # VK_LMENU
    0xA5,  # VK_RMENU
    0x5B,  # VK_LWIN
    0x5C,  # VK_RWIN
}


def _is_vk_pressed(vk_code: int) -> bool:
    if _user32 is None:
        return False
    return bool(_user32.GetAsyncKeyState(vk_code) & 0x8000)


def _get_pressed_vk_codes() -> set[int]:
    return {vk for vk in CAPTURE_VK_CODES if _is_vk_pressed(vk)}


class KeyCaptureDialog(QDialog):
    """
    PyQt6 對話框用於捕獲按鍵輸入（支援組合鍵）。
    
    Args:
        parent: 父視窗
        allow_combo: 是否允許組合鍵（True 則等待所有按鍵放開後確認）
        title: 對話框標題
    """
    
    def __init__(self, parent=None, allow_combo=False, title="按鍵偵測"):
        super().__init__(parent)
        self.allow_combo = allow_combo
        self.captured_keys = set()
        self.result_keys = []
        self._last_pressed_keys = set()
        self._combo_keys = set()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_input)
        
        self._setup_ui(title)
        self._start_capture_loop()
    
    def _setup_ui(self, title):
        """設置對話框 UI"""
        uic.loadUi(resource_path("ui/key_capture_dialog.ui"), self)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(300, 120)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        if self.allow_combo:
            hint_text = "請按下組合鍵...\n(放開所有按鍵後自動確認)\n按 ESC 取消"
        else:
            hint_text = "請按下任意按鍵...\n按 ESC 取消"

        self.hint_label.setText(hint_text)
        self.key_display_label.setText("")
    
    def _start_capture_loop(self):
        """啟動按鍵輪詢"""
        self._last_pressed_keys = _get_pressed_vk_codes()
        self._combo_keys.clear()
        self._poll_timer.start()

    def _stop_capture_loop(self):
        """停止按鍵輪詢"""
        if self._poll_timer.isActive():
            self._poll_timer.stop()

    def _poll_input(self):
        pressed_keys = _get_pressed_vk_codes()

        if VK_ESCAPE in pressed_keys:
            self.reject()
            return

        if self.allow_combo:
            self._poll_combo_keys(pressed_keys)
        else:
            self._poll_single_key(pressed_keys)

        self._last_pressed_keys = pressed_keys

    def _poll_single_key(self, pressed_keys):
        new_keys = pressed_keys - self._last_pressed_keys
        if not new_keys:
            return

        vk = sorted(new_keys)[0]
        self.result_keys = [vk]
        self._update_display(self.result_keys)
        self.accept()

    def _poll_combo_keys(self, pressed_keys):
        if pressed_keys:
            self.captured_keys = set(pressed_keys)
            self._combo_keys.update(pressed_keys)
            self._update_display(sorted(self.captured_keys))
            return

        if self._combo_keys and self._last_pressed_keys:
            self.result_keys = sorted(self._combo_keys)
            self._update_display(self.result_keys)
            self.accept()

    def _update_display(self, vk_codes):
        if not vk_codes:
            self.key_display_label.setText("")
            return

        key_names = [get_key_name_vk(hex(vk)) for vk in vk_codes]
        self.key_display_label.setText(" + ".join(key_names))
    
    def get_result(self):
        """獲取捕獲的按鍵結果（VK_CODE 名稱格式）"""
        if not self.result_keys:
            return None
        
        key_names = [get_key_name_vk(hex(vk)) for vk in self.result_keys]
        # 過濾掉 UNKNOWN
        key_names = [name for name in key_names if name != "UNKNOWN"]
        
        if not key_names:
            return None
        
        return "+".join(key_names)
    
    def closeEvent(self, event):
        """關閉事件處理"""
        self._stop_capture_loop()
        super().closeEvent(event)
    
    def reject(self):
        """取消對話框"""
        self._stop_capture_loop()
        self.result_keys = []
        super().reject()
    
    def accept(self):
        """確認對話框"""
        self._stop_capture_loop()
        super().accept()


def capture_key_with_dialog(parent=None, allow_combo=False, title="按鍵偵測"):
    """
    顯示按鍵捕獲對話框並返回結果。
    
    Args:
        parent: 父視窗
        allow_combo: 是否允許組合鍵
        title: 對話框標題
    
    Returns:
        str: 捕獲的 VK_CODE 名稱（組合鍵以 + 連接），如果取消則返回 None
    """
    dialog = KeyCaptureDialog(parent, allow_combo, title)
    result = dialog.exec()
    
    if result == QDialog.DialogCode.Accepted:
        return dialog.get_result()
    return None

class KeyCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Key Capture")
        self.label = tk.Label(
            root, text="Press any key...", font=('Helvetica', 16))
        self.label.pack(pady=20)

        # 设置焦点以便立即接收键盘事件
        self.root.focus_set()
        self._poll_job = None
        self._last_pressed_keys = set()
        self.event_result = None

        # 绑定关闭事件以确保正确清理资源
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_listening(self):
        self._last_pressed_keys = _get_pressed_vk_codes()
        self._poll_input()

    def _poll_input(self):
        pressed_keys = _get_pressed_vk_codes()

        if VK_ESCAPE in pressed_keys:
            self.event_result = None
            self.on_close()
            return

        new_keys = pressed_keys - self._last_pressed_keys
        if new_keys:
            self.event_result = hex(sorted(new_keys)[0])
            self.on_close()
            return

        self._last_pressed_keys = pressed_keys
        self._poll_job = self.root.after(POLL_INTERVAL_MS, self._poll_input)

    def on_close(self):
        if self._poll_job is not None:
            self.root.after_cancel(self._poll_job)
            self._poll_job = None
        self.root.quit()  # 使用 quit 而不是 destroy 以便返回结果

    def capture_event(self):
        self.start_listening()
        self.root.mainloop()
        return self.event_result


def get_keyboard_event(text="UNKNOWN"):
    result_queue = queue.Queue()

    def capture_key():
        key = KeyCaptureApp(tk.Tk()).capture_event()
        result_queue.put(key)

    thread = threading.Thread(target=capture_key)
    thread.start()
    # 等待线程完成
    thread.join()

    # 获取结果
    key = 'UNKNOWN' if result_queue.empty() else result_queue.get()

    return text if key is None or key == 'UNKNOWN' else key


keys_maps_vk = {
    "VK_LBUTTON": 0x01,
    "VK_RBUTTON": 0x02,
    "VK_CANCEL": 0x03,
    "VK_MBUTTON": 0x04,
    "VK_XBUTTON1": 0x05,
    "VK_XBUTTON2": 0x06,
    "VK_BACK": 0x08,
    "VK_TAB": 0x09,
    "VK_CLEAR": 0x0C,
    "VK_RETURN": 0x0D,
    "VK_SHIFT": 0x10,
    "VK_CONTROL": 0x11,
    "VK_MENU": 0x12,
    "VK_PAUSE": 0x13,
    "VK_CAPITAL": 0x14,
    "VK_KANA": 0x15,
    "VK_HANGUL": 0x15,
    "VK_IME_ON": 0x16,
    "VK_JUNJA": 0x17,
    "VK_FINAL": 0x18,
    "VK_HANJA": 0x19,
    "VK_KANJI": 0x19,
    "VK_IME_OFF": 0x1A,
    "VK_ESCAPE": 0x1B,
    "VK_CONVERT": 0x1C,
    "VK_NONCONVERT": 0x1D,
    "VK_ACCEPT": 0x1E,
    "VK_MODECHANGE": 0x1F,
    "VK_SPACE": 0x20,
    "VK_PRIOR": 0x21,
    "VK_NEXT": 0x22,
    "VK_END": 0x23,
    "VK_HOME": 0x24,
    "VK_LEFT": 0x25,
    "VK_UP": 0x26,
    "VK_RIGHT": 0x27,
    "VK_DOWN": 0x28,
    "VK_SELECT": 0x29,
    "VK_PRINT": 0x2A,
    "VK_EXECUTE": 0x2B,
    "VK_SNAPSHOT": 0x2C,
    "VK_INSERT": 0x2D,
    "VK_DELETE": 0x2E,
    "VK_HELP": 0x2F,
    "VK_0": 0x30,
    "VK_1": 0x31,
    "VK_2": 0x32,
    "VK_3": 0x33,
    "VK_4": 0x34,
    "VK_5": 0x35,
    "VK_6": 0x36,
    "VK_7": 0x37,
    "VK_8": 0x38,
    "VK_9": 0x39,
    "VK_A": 0x41,
    "VK_B": 0x42,
    "VK_C": 0x43,
    "VK_D": 0x44,
    "VK_E": 0x45,
    "VK_F": 0x46,
    "VK_G": 0x47,
    "VK_H": 0x48,
    "VK_I": 0x49,
    "VK_J": 0x4A,
    "VK_K": 0x4B,
    "VK_L": 0x4C,
    "VK_M": 0x4D,
    "VK_N": 0x4E,
    "VK_O": 0x4F,
    "VK_P": 0x50,
    "VK_Q": 0x51,
    "VK_R": 0x52,
    "VK_S": 0x53,
    "VK_T": 0x54,
    "VK_U": 0x55,
    "VK_V": 0x56,
    "VK_W": 0x57,
    "VK_X": 0x58,
    "VK_Y": 0x59,
    "VK_Z": 0x5A,
    "VK_LWIN": 0x5B,
    "VK_RWIN": 0x5C,
    "VK_APPS": 0x5D,
    "VK_SLEEP": 0x5F,
    "VK_NUMPAD0": 0x60,
    "VK_NUMPAD1": 0x61,
    "VK_NUMPAD2": 0x62,
    "VK_NUMPAD3": 0x63,
    "VK_NUMPAD4": 0x64,
    "VK_NUMPAD5": 0x65,
    "VK_NUMPAD6": 0x66,
    "VK_NUMPAD7": 0x67,
    "VK_NUMPAD8": 0x68,
    "VK_NUMPAD9": 0x69,
    "VK_MULTIPLY": 0x6A,
    "VK_ADD": 0x6B,
    "VK_SEPARATOR": 0x6C,
    "VK_SUBTRACT": 0x6D,
    "VK_DECIMAL": 0x6E,
    "VK_DIVIDE": 0x6F,
    "VK_F1": 0x70,
    "VK_F2": 0x71,
    "VK_F3": 0x72,
    "VK_F4": 0x73,
    "VK_F5": 0x74,
    "VK_F6": 0x75,
    "VK_F7": 0x76,
    "VK_F8": 0x77,
    "VK_F9": 0x78,
    "VK_F10": 0x79,
    "VK_F11": 0x7A,
    "VK_F12": 0x7B,
    "VK_F13": 0x7C,
    "VK_F14": 0x7D,
    "VK_F15": 0x7E,
    "VK_F16": 0x7F,
    "VK_F17": 0x80,
    "VK_F18": 0x81,
    "VK_F19": 0x82,
    "VK_F20": 0x83,
    "VK_F21": 0x84,
    "VK_F22": 0x85,
    "VK_F23": 0x86,
    "VK_F24": 0x87,
    "VK_NUMLOCK": 0x90,
    "VK_SCROLL": 0x91,
    "VK_LSHIFT": 0xA0,
    "VK_RSHIFT": 0xA1,
    "VK_LCONTROL": 0xA2,
    "VK_RCONTROL": 0xA3,
    "VK_LMENU": 0xA4,
    "VK_RMENU": 0xA5,
    "VK_BROWSER_BACK": 0xA6,
    "VK_BROWSER_FORWARD": 0xA7,
    "VK_BROWSER_REFRESH": 0xA8,
    "VK_BROWSER_STOP": 0xA9,
    "VK_BROWSER_SEARCH": 0xAA,
    "VK_BROWSER_FAVORITES": 0xAB,
    "VK_BROWSER_HOME": 0xAC,
    "VK_VOLUME_MUTE": 0xAD,
    "VK_VOLUME_DOWN": 0xAE,
    "VK_VOLUME_UP": 0xAF,
    "VK_MEDIA_NEXT_TRACK": 0xB0,
    "VK_MEDIA_PREV_TRACK": 0xB1,
    "VK_MEDIA_STOP": 0xB2,
    "VK_MEDIA_PLAY_PAUSE": 0xB3,
    "VK_LAUNCH_MAIL": 0xB4,
    "VK_LAUNCH_MEDIA_SELECT": 0xB5,
    "VK_LAUNCH_APP1": 0xB6,
    "VK_LAUNCH_APP2": 0xB7,
    "VK_OEM_1": 0xBA,
    "VK_OEM_PLUS": 0xBB,
    "VK_OEM_COMMA": 0xBC,
    "VK_OEM_MINUS": 0xBD,
    "VK_OEM_PERIOD": 0xBE,
    "VK_OEM_2": 0xBF,
    "VK_OEM_3": 0xC0,
    "VK_OEM_4": 0xDB,
    "VK_OEM_5": 0xDC,
    "VK_OEM_6": 0xDD,
    "VK_OEM_7": 0xDE,
    "VK_OEM_8": 0xDF,
    "VK_OEM_102": 0xE2,
    "VK_PROCESSKEY": 0xE5,
    "VK_PACKET": 0xE7,
    "VK_ATTN": 0xF6,
    "VK_CRSEL": 0xF7,
    "VK_EXSEL": 0xF8,
    "VK_EREOF": 0xF9,
    "VK_PLAY": 0xFA,
    "VK_ZOOM": 0xFB,
    "VK_PA1": 0xFD,
    "VK_OEM_CLEAR": 0xFE
}


def get_key_name_vk(hex):
    return dict(
        zip(keys_maps_vk.values(),
            keys_maps_vk.keys())
    ).get(int(hex, 16), "UNKNOWN")


def get_key_code_vk(key_name):
    return keys_maps_vk.get(key_name, "UNKNOWN")


class KeyboardListener:
    """
    使用 GetAsyncKeyState 輪詢鍵盤與滑鼠狀態，避免全域 Hook。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 防止重複初始化
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._listener_lock = threading.Lock()
        self._is_running_event = threading.Event()
        self._initialized = True

    def start(self):
        with self._listener_lock:
            self._is_running_event.set()

    def stop(self):
        with self._listener_lock:
            self._is_running_event.clear()

    def is_pressed(self, vk_code: int) -> bool:
        """
        檢查指定的虛擬鍵碼 (VK code) 是否被按下。
        """
        if not self._is_running_event.is_set():
            return False

        if vk_code is None:
            return False

        try:
            vk_code = int(vk_code)
        except (TypeError, ValueError):
            return False

        return _is_vk_pressed(vk_code)

# 建立一個全域單例
keyboard_listener = KeyboardListener()
