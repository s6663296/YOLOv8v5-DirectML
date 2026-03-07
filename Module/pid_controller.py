"""
PID 控制器滑鼠移動模組
使用 PID 控制器實現平滑、可調的滑鼠追蹤軌跡
取代原有的貝茲曲線模組
"""
import time
from typing import Tuple


class PIDController:
    """
    單軸 PID 控制器
    
    根據誤差（目標偏移量）動態計算控制輸出：
    - P（比例）：立即響應當前誤差
    - I（積分）：消除長期穩態誤差
    - D（微分）：抑制震盪和超調
    """
    
    def __init__(self, kp: float = 0.5, ki: float = 0.0, kd: float = 0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # 內部狀態
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_update = True
        
        # 積分限幅（anti-windup），防止積分項過大
        self._integral_limit = 50.0
        
    def set_gains(self, kp: float, ki: float, kd: float):
        """設置 PID 增益參數"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
    def update(self, error: float, dt: float) -> float:
        """
        根據當前誤差計算 PID 控制量
        
        Args:
            error: 當前誤差（目標值 - 當前值）
            dt: 距上次更新的時間差（秒）
            
        Returns:
            PID 控制輸出值
        """
        if dt <= 0:
            return error * self.kp
        
        # 方向跳變偵測：如果誤差方向反轉，清除積分項
        # 這防止了目標改變方向時的「慣性衝過頭」
        if self._prev_error != 0.0 and not self._first_update:
            if (error > 0 and self._prev_error < 0) or (error < 0 and self._prev_error > 0):
                self._integral = 0.0
        
        # P 項
        p_term = self.kp * error
        
        # I 項（含 anti-windup 限幅）
        self._integral += error * dt
        self._integral = max(-self._integral_limit, 
                           min(self._integral_limit, self._integral))
        i_term = self.ki * self._integral
        
        # D 項
        if self._first_update:
            d_term = 0.0
            self._first_update = False
        else:
            derivative = (error - self._prev_error) / dt
            d_term = self.kd * derivative
        
        self._prev_error = error
        
        return p_term + i_term + d_term
    
    def reset(self):
        """重置 PID 控制器狀態"""
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_update = True


class PIDMouseMover:
    """
    PID 滑鼠移動控制器
    管理 X/Y 雙軸 PID 控制並與瞄準邏輯整合
    取代原有的 BezierMouseMover
    """
    
    def __init__(self):
        self.enabled = False
        self._pid_x = PIDController()
        self._pid_y = PIDController()
        self._last_time = 0.0
        self._first_frame = True
        
    def set_parameters(self, enabled: bool, kp: float = 0.5,
                       ki: float = 0.0, kd: float = 0.01):
        """
        設置 PID 參數
        
        Args:
            enabled: 是否啟用 PID 平滑
            kp: 比例增益 (0.0-1.0)
            ki: 積分增益 (0.0-0.05)
            kd: 微分增益 (0.0-0.1)
        """
        self.enabled = enabled
        self._pid_x.set_gains(kp, ki, kd)
        self._pid_y.set_gains(kp, ki, kd)
        
        if not enabled:
            self.reset()
            
    def update(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """
        根據目標移動量計算 PID 平滑後的移動量
        
        Args:
            target_x: 計算出的目標 X 移動量
            target_y: 計算出的目標 Y 移動量
            
        Returns:
            PID 平滑後的移動量 (x, y)
        """
        if not self.enabled:
            return (target_x, target_y)
        
        current_time = time.perf_counter()
        
        if self._first_frame:
            self._last_time = current_time
            self._first_frame = False
            # 第一幀直接使用 P 項響應
            return (target_x * self._pid_x.kp, target_y * self._pid_y.kp)
        
        dt = current_time - self._last_time
        self._last_time = current_time
        
        # 防止 dt 異常（過大或過小）
        dt = max(0.001, min(dt, 0.1))
        
        # PID 控制：誤差就是目標移動量（因為我們希望把偏差歸零）
        move_x = self._pid_x.update(target_x, dt)
        move_y = self._pid_y.update(target_y, dt)
        
        return (move_x, move_y)
    
    def reset(self):
        """重置 PID 控制器狀態"""
        self._pid_x.reset()
        self._pid_y.reset()
        self._first_frame = True
        self._last_time = 0.0
