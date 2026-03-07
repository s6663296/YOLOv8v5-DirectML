import win32api
import win32con
import ctypes
import win32com.client
import threading
from Module.logger import logger

# 驅動程式的全域變數
kmdll = None
km_driver_is_ready = False

# hid 驅動程式的全域變數
hiddll = None
hid_driver_is_ready = False

# 按鍵魔盒-Lite 驅動程式的全域變數
kmbox_lite_dll = None
kmbox_lite_driver_is_ready = False
 
# Makcu 驅動程式的全域變數
makcu_controller = None
makcu_driver_is_ready = False
MAKCU_HW_LOCK = threading.Lock()
 
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


def init_kmbox_lite_driver():
    """
    嘗試初始化 按鍵魔盒-Lite 驅動程式。
    """
    global kmbox_lite_dll, kmbox_lite_driver_is_ready
    try:
        kmbox_lite_dll = win32com.client.Dispatch("kmdll.KM")
        if kmbox_lite_dll.OpenDevice:
            logger.info("按鍵魔盒-Lite 設備已連接成功")
            kmbox_lite_driver_is_ready = True
        else:
            logger.warning("按鍵魔盒-Lite 設備未連接或驅動加載失敗。")
            kmbox_lite_driver_is_ready = False
    except Exception as e:
        logger.error(f"加載 按鍵魔盒-Lite COM 組件或連接設備失敗: {e}")
        kmbox_lite_dll = None
        kmbox_lite_driver_is_ready = False

def is_kmbox_lite_driver_ready():
    """返回 按鍵魔盒-Lite 驅動程式的狀態。"""
    return kmbox_lite_driver_is_ready
 
def init_makcu_driver():
    """
    嘗試初始化 Makcu 驅動程式。
    成功返回 True，失敗返回 False。
    """
    global makcu_controller, makcu_driver_is_ready
    try:
        from makcu import create_controller
        with MAKCU_HW_LOCK:
            # 如果控制器已存在，先斷開連接
            if makcu_controller:
                try:
                    logger.info("偵測到已存在的 Makcu 控制器，正在斷開連線...")
                    makcu_controller.disconnect()
                    makcu_controller = None
                    logger.info("舊的 Makcu 控制器連線已成功斷開。")
                except Exception as e:
                    logger.error(f"斷開舊的 Makcu 控制器時發生錯誤: {e}")
            
            logger.info("正在初始化 Makcu 控制器...")
            makcu_controller = create_controller(debug=False, auto_reconnect=True)
            makcu_controller.enable_button_monitoring(True)
        logger.info("Makcu 控制器已連接並啟用按鍵監測。")
        makcu_driver_is_ready = True
        return True
    except ImportError:
        logger.error("缺少 'makcu' 函式庫。請執行 'pip install makcu' 進行安裝。")
        makcu_controller = None
        makcu_driver_is_ready = False
        return False
    except Exception as e:
        logger.error(f"初始化 Makcu 控制器失敗: {e}")
        makcu_controller = None
        makcu_driver_is_ready = False
        return False

def is_makcu_driver_ready():
    """返回 Makcu 驅動程式的狀態。"""
    return makcu_driver_is_ready and makcu_controller is not None
 
def move(mode, centerx, centery):
    """根據指定的模式移動滑鼠"""
    match mode:
        case "按鍵魔盒-鍵鼠轉接器":
            if km_driver_is_ready:
                kmdll.MoveR(int(centerx), int(centery))
        case "hid007":
            if hid_driver_is_ready:
                hiddll.move(int(centerx), int(centery))
        case "按鍵魔盒-Lite":
            if not kmbox_lite_driver_is_ready:
                init_kmbox_lite_driver()
            if kmbox_lite_driver_is_ready:
                # 使用 COM 介面移動滑鼠，event 9 為相對移動
                kmbox_lite_dll.mouse_event(9, int(centerx), int(centery))
        case "Makcu":
            # 每次都檢查驅動狀態，如果未就緒則嘗試初始化
            if not is_makcu_driver_ready():
                if not init_makcu_driver():
                    # 如果初始化失敗，則直接返回，不執行後續操作
                    return
            
            # 再次確認驅動就緒後，執行移動
            if is_makcu_driver_ready():
                with MAKCU_HW_LOCK:
                    makcu_controller.move(int(centerx), int(centery))
        case "win32":
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(centerx), int(centery), 0, 0)
