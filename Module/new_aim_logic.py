import time
import math
import win32api
import win32con

# Import necessary components from the existing project
from Module.config import Config
from Module.logger import logger
import Module.control as control

class NewAimLogic:
    def __init__(self, app_instance):
        """
        初始化新的瞄準邏輯控制器。
        Args:
            app_instance: 主應用程式實例 (SudaneseboyApp)。
        """
        self.app = app_instance
        self._initialize_state()
        self.update_parameters()
        self.setup_hardware()

    def _initialize_state(self):
        """初始化不應在執行時重設的狀態變數。"""
        self.prev_x = 0
        self.prev_y = 0
        self.prev_time = None
        self.prev_velocity_x = 0
        self.prev_velocity_y = 0
        self.prev_distance = None
        self.last_move_x = 0
        self.last_move_y = 0
        self.bScope = False

    def update_parameters(self):
        """
        從應用程式實例更新所有可變參數。
        當設定或擷取大小變更時應呼叫此函式。
        """
        # 螢幕相關參數
        self.screen_width = self.app.capture_area["width"]
        self.screen_height = self.app.capture_area["height"]
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        logger.info(f"為螢幕大小更新了 AimLogic 參數: {self.screen_width}x{self.screen_height}")

        # 瞄準設定
        self.aim_speed_x = self.app.aim_speed_x
        self.aim_speed_y = self.app.aim_speed_y
        
        # 預測設定
        self.prediction_enabled = self.app.prediction_enabled
        self.prediction_interval = self.app.prediction_time
        self.speed_correction_factor = 0.1
        
        # 平滑因子
        self.smoothing_factor = 0.85
        
        # 動態速度乘數
        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2) / 2
        self.min_speed_multiplier = 0.7
        self.max_speed_multiplier = 1.2
        self.auto_lock_mode = self.app.auto_lock_mode

    def setup_hardware(self):
        """如果需要，可以擴展此功能以支援 ghub、rzr 等。"""
        pass

    def process_data(self, detection_result):
        """
        單一偵測的主要處理函式。
        Args:
            detection_result (dict): 包含 'center_x'、'center_y'、'box_width'、'box_height' 的字典。
        """
        if not detection_result:
            self.reset_state()
            return

        target_x, target_y = detection_result['center_x'], detection_result['center_y']
        box_w, box_h = detection_result['box_width'], detection_result['box_height']
        
        # 1. 預測未來位置
        if self.prediction_enabled:
            current_time = time.time()
            predicted_x, predicted_y = self.predict_target_position(target_x, target_y, current_time)
        else:
            predicted_x, predicted_y = target_x, target_y

        # 2. 根據預測位置計算移動
        move_x, move_y = self.calc_movement(predicted_x, predicted_y, box_w, box_h)
        
        # 3. 移動滑鼠
        self.move_mouse(move_x, move_y)

    def predict_target_position(self, target_x, target_y, current_time):
        if self.prev_time is None:
            self.prev_time = current_time
            self.prev_x, self.prev_y = target_x, target_y
            self.prev_velocity_x, self.prev_velocity_y = 0, 0
            return target_x, target_y

        delta_time = current_time - self.prev_time
        if delta_time < 1e-6:
            return target_x, target_y # 避免除以零

        # 檢查目標跳動（新目標）
        max_jump = self.screen_width * 0.3
        if abs(target_x - self.prev_x) > max_jump or abs(target_y - self.prev_y) > max_jump:
            self.reset_state()
            self.prev_x, self.prev_y = target_x, target_y
            self.prev_time = current_time
            return target_x, target_y

        velocity_x = (target_x - self.prev_x) / delta_time
        velocity_y = (target_y - self.prev_y) / delta_time
        
        acceleration_x = (velocity_x - self.prev_velocity_x) / delta_time
        acceleration_y = (velocity_y - self.prev_velocity_y) / delta_time

        prediction_interval = delta_time * self.prediction_interval
        
        predicted_x = target_x + velocity_x * prediction_interval + 0.5 * acceleration_x * (prediction_interval ** 2)
        predicted_y = target_y + velocity_y * prediction_interval + 0.5 * acceleration_y * (prediction_interval ** 2)

        self.prev_x, self.prev_y = target_x, target_y
        self.prev_velocity_x, self.prev_velocity_y = velocity_x, velocity_y
        self.prev_time = current_time
        
        return predicted_x, predicted_y

    def calculate_speed_multiplier(self, distance):
        """根據與中心的距離計算速度乘數。"""
        if self.max_distance == 0: return 1.0
        
        normalized_distance = min(distance / self.max_distance, 1)
        speed_multiplier = self.min_speed_multiplier + (self.max_speed_multiplier - self.min_speed_multiplier) * normalized_distance
        return speed_multiplier

    def calc_movement(self, target_x, target_y, box_width, box_height):
        offset_centerx_ratio = self.app.offset_centerx
        offset_centery_ratio = self.app.offset_centery
        
        pixel_offset_x = offset_centerx_ratio * (box_width / 2)
        pixel_offset_y = offset_centery_ratio * (box_height / 2)

        adjusted_target_x = target_x + pixel_offset_x
        adjusted_target_y = target_y + pixel_offset_y
        
        offset_x = adjusted_target_x - self.center_x
        offset_y = adjusted_target_y - self.center_y
        
        distance = math.sqrt(offset_x**2 + offset_y**2)
        
        speed_multiplier = self.calculate_speed_multiplier(distance)
        
        move_x = self.smoothing_factor * offset_x + (1 - self.smoothing_factor) * self.last_move_x
        move_y = self.smoothing_factor * offset_y + (1 - self.smoothing_factor) * self.last_move_y
        
        self.last_move_x, self.last_move_y = move_x, move_y

        final_move_x = move_x * self.aim_speed_x / 10.0 * speed_multiplier
        final_move_y = move_y * self.aim_speed_y / 10.0 * speed_multiplier
        
        return final_move_x, final_move_y

    def move_mouse(self, x, y):
        if abs(x) < 1 and abs(y) < 1:
            return
            
        move_mode = Config.get("mouse_move_mode", "win32")
        use_threading = Config.get("mouse_threading", True)
        
        lock_key_pressed = self.is_lock_key_pressed()

        if self.app.aimbot_enabled and lock_key_pressed:
            if use_threading:
                self.app.thread_pool_manager.submit_mouse_move(move_mode, int(x), int(y))
            else:
                control.move(move_mode, int(x), int(y))
        else:
             self.reset_state()


    def is_lock_key_pressed(self):
        """檢查主要或側邊鎖定鍵是否被按下。"""
        if self.auto_lock_mode:
            return True
            
        primary_key = Config.get("lockKey", "VK_RBUTTON")
        side_key_enabled = Config.get("side_key_lock_enabled", False)
        side_key = Config.get("side_lock_key", "VK_XBUTTON1")

        primary_vk_code = getattr(win32con, primary_key, None)
        if primary_vk_code and win32api.GetAsyncKeyState(primary_vk_code) & 0x8000:
            return True
            
        if side_key_enabled:
            side_vk_code = getattr(win32con, side_key, None)
            if side_vk_code and win32api.GetAsyncKeyState(side_vk_code) & 0x8000:
                return True
        
        return False

    def reset_state(self):
        """重設預測器和平滑器的狀態。"""
        self.prev_time = None
        self.prev_x = 0
        self.prev_y = 0
        self.prev_velocity_x = 0
        self.prev_velocity_y = 0
        self.last_move_x = 0
        self.last_move_y = 0