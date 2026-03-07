import time
import math
from collections import deque

# Import necessary components from the existing project
from Module.config import Config
from Module.logger import logger
import Module.control as control
from Module.keyboard import keyboard_listener, get_key_code_vk
from Module.pid_controller import PIDMouseMover

class NewAimLogic:
    def __init__(self, app_instance):
        """
        初始化新的瞄準邏輯控制器。
        """
        self.app = app_instance
        self._initialize_state()
        
        # 初始化 PID 移動控制器
        self.pid_mover = PIDMouseMover()
        
        self.update_parameters()
        self.setup_hardware()

    def _initialize_state(self):
        """初始化不應在執行時重設的狀態變數。"""
        self.prev_x = 0
        self.prev_y = 0
        self.prev_time = None
        self.prev_distance = None
        self.last_move_x = 0
        self.last_move_y = 0
        self.bScope = False
        # 追蹤計算後的目標是否在瞄準範圍內（用於 overlay 顏色變化）
        self.target_in_aim_range = False
        # --- 預測狀態（TensorRT 風格：中位數濾波 + 一致性加權線性預測）---
        self._velocity_x = 0.0  # 平滑後的速度
        self._velocity_y = 0.0
        self._velocities = deque(maxlen=3)  # 最近速度歷史（3幀：平衡抗抖動與反應速度）
        
        # --- 目標切換偵測 ---
        self._prev_target_cx = None  # 上一幀的目標中心 X（用於偵測目標切換）
        self._prev_target_cy = None
        
        # --- 滑鼠移動小數點累積 (Residual Accumulation) ---
        self._accumulator_x = 0.0
        self._accumulator_y = 0.0

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
        self.aim_range = self.app.aim_range # 保留 aim_range 以防萬一，但主要使用 width/height
        self.aim_width = self.app.aim_width
        self.aim_height = self.app.aim_height
        
        # 直接使用 width/height 作為有效瞄準範圍
        self.effective_aim_width = self.aim_width
        self.effective_aim_height = self.aim_height
        self.aim_speed_x = self.app.aim_speed_x
        self.aim_speed_y = self.app.aim_speed_y
        self.offset_mode = self.app.offset_mode
        self.offset_centerx_ratio = self.app.offset_centerx_ratio
        self.offset_centery_ratio = self.app.offset_centery_ratio
        self.offset_centerx_pixel = self.app.offset_centerx_pixel
        self.offset_centery_pixel = self.app.offset_centery_pixel

        # 側鍵鎖定偏移設定
        self.exclude_side_key_offset = getattr(self.app, 'exclude_side_key_offset', False)
        self.side_key_enabled = getattr(self.app, 'side_key_lock_enabled', False)
        side_key_name = getattr(self.app, 'side_lock_key', "VK_XBUTTON1")
        self.side_vk_code = get_key_code_vk(side_key_name)
        
        # 預測設定
        self.prediction_enabled = self.app.prediction_enabled
        self.prediction_interval = self.app.prediction_time
        self.auto_lock_mode = self.app.auto_lock_mode
        

        
        # PID 控制器設定
        pid_enabled = getattr(self.app, 'pid_enabled', False)
        # UI 中的 pid_kp 是 0-100 的整數，轉換為 0.0-1.0
        pid_kp = getattr(self.app, 'pid_kp', 50) / 100.0
        # UI 中的 pid_ki 是 0-50 的整數，轉換為 0.0-0.05
        pid_ki = getattr(self.app, 'pid_ki', 0) / 1000.0
        # UI 中的 pid_kd 是 0-100 的整數，轉換為 0.0-0.1
        pid_kd = getattr(self.app, 'pid_kd', 10) / 1000.0
        
        self.pid_mover.set_parameters(
            enabled=pid_enabled,
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd
        )


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
        
        # 0. 偵測目標切換（偵測框中心跳變 > 框大小的 1.5 倍）
        target_switched = False
        if self._prev_target_cx is not None:
            switch_threshold = max(box_w, box_h) * 1.5
            dx = abs(target_x - self._prev_target_cx)
            dy = abs(target_y - self._prev_target_cy)
            if dx > switch_threshold or dy > switch_threshold:
                target_switched = True
                logger.debug(f"目標切換偵測: dx={dx:.1f}, dy={dy:.1f}, threshold={switch_threshold:.1f}")
        
        # 記錄當前目標中心（用於下一幀的切換偵測）
        self._prev_target_cx = target_x
        self._prev_target_cy = target_y
        
        # 如果目標切換了，重設預測歷史和 PID 狀態
        if target_switched:
            self._reset_prediction_state()
            self.pid_mover.reset()
        
        # 1. 預測未來位置
        if self.prediction_enabled:
            current_time = time.perf_counter()
            predicted_x, predicted_y = self.predict_target_position(target_x, target_y, current_time)
        else:
            predicted_x, predicted_y = target_x, target_y

        # 2. 根據預測位置計算移動（持續追蹤模型範圍內最近的目標）
        move_x, move_y, distance = self.calc_movement(predicted_x, predicted_y, box_w, box_h)
        
        # 3. 更新目標是否在瞄準範圍內的狀態（用於 overlay 顏色變化）
        # Square/Rectangle check
        in_width = abs(move_x) <= (self.effective_aim_width / 2)
        in_height = abs(move_y) <= (self.effective_aim_height / 2)
        self.target_in_aim_range = in_width and in_height
        
        # 4. 移動滑鼠（只有當目標在瞄準範圍內才實際鎖定）
        self.move_mouse(move_x, move_y, self.target_in_aim_range)

    def predict_target_position(self, target_x, target_y, current_time):
        """
        使用中位數濾波速度估計 + 一致性加權線性預測。
        （參考 TensorRT.py 的預測方式）
        
        - 中位數濾波比均值更能抵抗偵測框跳動的離群值
        - 預測權重根據速度一致性（方差）動態調整：
          目標移動穩定 → 預測更大膽，目標亂動 → 減少預測
        - 速度上限防止過沖
        """
        if self.prev_time is None:
            self.prev_time = current_time
            self.prev_x, self.prev_y = target_x, target_y
            self._velocity_x = 0.0
            self._velocity_y = 0.0
            self._velocities.clear()
            return target_x, target_y

        delta_time = current_time - self.prev_time
        if delta_time < 1e-6:
            return target_x, target_y  # 避免除以零

        # 檢查目標跳動（新目標出現，重設狀態）
        max_jump = self.screen_width * 0.3
        if abs(target_x - self.prev_x) > max_jump or abs(target_y - self.prev_y) > max_jump:
            self._reset_prediction_state()
            self.prev_x, self.prev_y = target_x, target_y
            self.prev_time = current_time
            return target_x, target_y

        # 計算瞬時速度（像素/秒）
        instant_vx = (target_x - self.prev_x) / delta_time
        instant_vy = (target_y - self.prev_y) / delta_time

        # 儲存速度歷史
        self._velocities.append((instant_vx, instant_vy))

        # --- 速度平滑 ---
        if len(self._velocities) >= 3:
            # 使用中位數濾波（比均值更能抵抗離群值）
            velx_list = [v[0] for v in self._velocities]
            vely_list = [v[1] for v in self._velocities]
            self._velocity_x = sorted(velx_list)[len(velx_list) // 2]
            self._velocity_y = sorted(vely_list)[len(vely_list) // 2]
        else:
            # 樣本不足時使用 EMA（70% 新值 + 30% 舊值）
            ema_alpha = 0.7
            self._velocity_x = ema_alpha * instant_vx + (1 - ema_alpha) * self._velocity_x
            self._velocity_y = ema_alpha * instant_vy + (1 - ema_alpha) * self._velocity_y

        # --- 線性預測（使用中位數濾波後的速度）---
        prediction_time = delta_time * self.prediction_interval
        predicted_x = target_x
        predicted_y = target_y

        # 速度上限防止過沖（像素/秒）
        max_velocity = 500.0
        if abs(self._velocity_x) < max_velocity and abs(self._velocity_y) < max_velocity:
            predicted_x += self._velocity_x * prediction_time
            predicted_y += self._velocity_y * prediction_time

        self.prev_x, self.prev_y = target_x, target_y
        self.prev_time = current_time

        return predicted_x, predicted_y



    def calc_movement(self, target_x, target_y, box_width, box_height):
        pixel_offset_x = 0
        pixel_offset_y = 0
        
        apply_offset = True
        # 如果啟用了排除側鍵偏移，且側鍵被按下，則不應用偏移
        if self.exclude_side_key_offset and self.side_key_enabled:
            if self.side_vk_code != "UNKNOWN" and keyboard_listener.is_pressed(self.side_vk_code):
                apply_offset = False
        
        if apply_offset:
            if self.offset_mode == "像素偏移":
                pixel_offset_x = self.offset_centerx_pixel
                pixel_offset_y = self.offset_centery_pixel
            else: # 比例偏移
                pixel_offset_x = self.offset_centerx_ratio * (box_width / 2)
                pixel_offset_y = self.offset_centery_ratio * (box_height / 2)

        adjusted_target_x = target_x + pixel_offset_x
        adjusted_target_y = target_y + pixel_offset_y
        
        offset_x = adjusted_target_x - self.center_x
        offset_y = adjusted_target_y - self.center_y
        
        distance = math.sqrt(offset_x**2 + offset_y**2)

        # 回傳原始偏移量（不乘 aim_speed），讓 PID 處理原始訊號
        # 這樣可以避免把噪聲放大後再給 PID，減少抖動
        return offset_x, offset_y, distance

    def move_mouse(self, x, y, in_range):
        if abs(x) < 1 and abs(y) < 1:
            return

        lock_key_pressed = self.is_lock_key_pressed()
        if not (self.app.aimbot_enabled and lock_key_pressed):
            # 即使不移動滑鼠，也不重置狀態，保持追蹤計算的連續性
            # 重置 PID 狀態
            self.pid_mover.reset()
            self._accumulator_x = 0.0
            self._accumulator_y = 0.0
            return

        # 檢查目標是否在瞄準範圍內（只有在瞄準範圍內才實際移動滑鼠）
        if self.app.aim_overlay_enabled and not in_range:
            # 目標在模型範圍內但超出瞄準範圍，不移動滑鼠
            self.pid_mover.reset()
            self._accumulator_x = 0.0
            self._accumulator_y = 0.0
            return

        # 應用 PID 控制器平滑移動（傳入原始偏移量）
        pid_out_x, pid_out_y = self.pid_mover.update(x, y)
        
        # PID 輸出後再乘上 aim_speed（速度/靈敏度調整）
        # 這樣做的好處是 PID 處理的是真實物理距離，不會被 speed 参数扭曲
        raw_move_x = pid_out_x * (self.aim_speed_x / 10.0)
        raw_move_y = pid_out_y * (self.aim_speed_y / 10.0)
        
        # --- 應用小數點累積 (Residual Accumulation) ---
        # 將當前計算出的浮點數移動量加入累積器
        total_x = raw_move_x + self._accumulator_x
        total_y = raw_move_y + self._accumulator_y
        
        # 取出整數部分作為本次實際移動量
        final_x_int = int(total_x)
        final_y_int = int(total_y)
        
        # 將剩餘的小數部分存回累積器，留待下一幀使用
        self._accumulator_x = total_x - final_x_int
        self._accumulator_y = total_y - final_y_int
        
        # 確保移動量仍然足夠 (這裡檢查的是整數移動量，如果為0就不移動)
        if final_x_int == 0 and final_y_int == 0:
            return

        move_mode = Config.get("mouse_move_mode", "win32")
        use_threading = Config.get("mouse_threading", True)
        
        if use_threading:
            self.app.thread_pool_manager.submit_mouse_move(move_mode, final_x_int, final_y_int, priority=1)
        else:
            control.move(move_mode, final_x_int, final_y_int)


    def is_lock_key_pressed(self):
        """使用 KeyboardListener 檢查主要或側邊鎖定鍵是否被按下。"""
        if self.auto_lock_mode:
            return True
    
        primary_key_name = Config.get("lockKey", "VK_RBUTTON")
        primary_vk_code = get_key_code_vk(primary_key_name)
        if primary_vk_code != "UNKNOWN" and keyboard_listener.is_pressed(primary_vk_code):
            return True
    
        side_key_enabled = Config.get("side_key_lock_enabled", False)
        if side_key_enabled:
            side_key_name = Config.get("side_lock_key", "VK_XBUTTON1")
            side_vk_code = get_key_code_vk(side_key_name)
            if side_vk_code != "UNKNOWN" and keyboard_listener.is_pressed(side_vk_code):
                return True
                
        return False

    def _reset_prediction_state(self):
        """只重設預測相關的狀態（速度歷史），不影響其他狀態。"""
        self.prev_time = None
        self.prev_x = 0
        self.prev_y = 0
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._velocities.clear()

    def reset_state(self):
        """重設預測器和平滑器的狀態。"""
        self._reset_prediction_state()
        self.last_move_x = 0
        self.last_move_y = 0
        self.target_in_aim_range = False
        self._prev_target_cx = None
        self._prev_target_cy = None
        # 重置 PID 狀態
        self.pid_mover.reset()
        self._accumulator_x = 0.0
        self._accumulator_y = 0.0
