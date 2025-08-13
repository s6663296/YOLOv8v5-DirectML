import mouse
import win32api
import win32con
import ctypes
import threading
import time
from Module.config import Config
from Module.logger import logger

# 驅動程式的全域變數
kmdll = None
km_driver_is_ready = False

# hid 驅動程式的全域變數
hiddll = None
hid_driver_is_ready = False

# dd 驅動程式的全域變數
dd_dll = None
dd_driver_is_ready = False
 
def init_km_driver():
    """
    嘗試初始化「按鍵魔盒-鍵鼠轉接器」驅動程式。
    根據成功或失敗設定全域旗標。
    """
    global kmdll, km_driver_is_ready
    try:
        kmdll = ctypes.CDLL(r'./dll64.dll')
        if kmdll.IsOpen():
            logger.info("按鍵魔盒-鍵鼠轉接器 設備已連接成功")
            km_driver_is_ready = True
        else:
            kmdll.OpenDeviceByID(0, 0)
            if kmdll.IsOpen():
                logger.info("按鍵魔盒-鍵鼠轉接器 設備已連接成功")
                km_driver_is_ready = True
            else:
                logger.warning("按鍵魔盒-鍵鼠轉接器 設備未連接或驅動加載失敗。")
                km_driver_is_ready = False
    except Exception as e:
        logger.error(f"加載 按鍵魔盒-鍵鼠轉接器 DLL 或連接設備失敗: {e}")
        kmdll = None
        km_driver_is_ready = False

def is_km_driver_ready():
    """返回驅動程式的狀態。"""
    return km_driver_is_ready

def init_hid_driver():
    """
    嘗試初始化 hid007 驅動程式。
    """
    global hiddll, hid_driver_is_ready
    try:
        hiddll = ctypes.cdll.LoadLibrary('./hiddll_x64.dll')
        if hiddll.open_hiddev_default() >= 0:
            logger.info("hid007 設備已連接成功")
            hid_driver_is_ready = True
        else:
            logger.warning("hid007 設備未連接或驅動加載失敗。")
            hid_driver_is_ready = False
    except Exception as e:
        logger.error(f"加載 hid007 DLL 或連接設備失敗: {e}")
        hiddll = None
        hid_driver_is_ready = False

def is_hid_driver_ready():
    """返回 hid 驅動程式的狀態。"""
    return hid_driver_is_ready

def init_dd_driver():
    """
    嘗試初始化 ddxoft 驅動程式。
    """
    global dd_dll, dd_driver_is_ready
    try:
        dd_dll = ctypes.windll.LoadLibrary(r'./ddxoft.dll')
        if dd_dll.DD_btn(0) == 1:
            logger.info("ddxoft 設備已連接成功")
            dd_driver_is_ready = True
        else:
            logger.warning("ddxoft 設備未連接或驅動加載失敗。")
            dd_driver_is_ready = False
    except Exception as e:
        logger.error(f"加載 ddxoft DLL 或連接設備失敗: {e}")
        dd_dll = None
        dd_driver_is_ready = False

def is_dd_driver_ready():
    """返回 dd 驅動程式的狀態。"""
    return dd_driver_is_ready

def move(mode, centerx, centery):
    """根據指定的模式移動滑鼠"""
    match mode:
        case "按鍵魔盒-鍵鼠轉接器":
            if km_driver_is_ready:
                kmdll.MoveR(int(centerx), int(centery))
        case "hid007":
            if hid_driver_is_ready:
                hiddll.move(int(centerx), int(centery))
        case "ddxoft":
            if not dd_driver_is_ready:
                init_dd_driver()
            if dd_driver_is_ready:
                dd_dll.DD_movR(int(centerx), int(centery))
        case "win32":
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(centerx), int(centery), 0, 0)
