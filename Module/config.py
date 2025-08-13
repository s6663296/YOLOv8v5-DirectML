import os
from pathlib import Path
import sys
from typing import Any
import json

Root = Path(os.path.realpath(sys.argv[0])).parent


class _Config:
    def __init__(self):
        self.default = {
            "log_level": "info",
            "aim_range": 150,
            "aimBot": True,
            "confidence": 0.3,
            "aim_speed_x": 6.7,
            "aim_speed_y": 8.3,
            "model_file": "yolov8n.pt",
            "model_size": 320,
            "mouse_Side_Button_Witch": True,
            "ProcessMode": "multi_process",
            "window_always_on_top": False,
            "target_class": "0",
            "lockKey": "VK_RBUTTON",
            "triggerType": "按下",
            "offset_centery": 0.75,
            "offset_centerx": 0.0,
            "screen_pixels_for_360_degrees": 6550,
            "screen_height_pixels": 3220,
            "near_speed_multiplier": 2.5,
            "slow_zone_radius": 0,
            "mouseMoveMode": "win32",
            "lockSpeed": 5.5,
            "jump_suppression_switch": False,
            "side_key_lock": False,
            "kalman_delay": 0.2,
            "auto_scale_aim_range": False,
            "auto_scale_factor": 0.0,
            "mouse_threading": True,
            "recoil_control_enabled": False,
            "recoil_y_strength": 5,
            "recoil_x_strength": 0,
            "recoil_delay": 0.01,
            "recoil_trigger_keys": "VK_LBUTTON",
            "overlay_color_r": 0,
            "overlay_color_g": 255,
            "overlay_color_b": 255,
            "overlay_color_a": 200,
            "overlay_lock_color_r": 255,
            "overlay_lock_color_g": 0,
            "overlay_lock_color_b": 0,
            "overlay_lock_color_a": 200,
            "show_fps_overlay": False,
            "show_detection_boxes": False,
            "capture_fps": 240,
            "yolo_version": "auto",
        }
        self.content = self.read()
        self.current_config_file = "settings.json"  # 追蹤當前配置文件

    def read(self, config_file: str = "settings.json") -> dict:
        try:
            os.makedirs(Root / "Data", exist_ok=True)
            with open(Root / "Data" / config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return self.default.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """
        獲取設定項的值，如果不存在則返回預設值。

        :param key: 設定項的鍵
        :param default: 預設值
        :return: 返回設定項的值
        """
        if default is not None:
            return self.content.get(key, default)
        return self.content.get(key, self.default.get(key))

    def update(self, key: str, value: Any) -> None:
        self.content[key] = value

    def delete(self, key: str) -> None:
        if key in self.content:
            del self.content[key]
            self.save()

    def save(self, config_file: str = None) -> None:
        """保存配置到指定文件，如果未指定則保存到當前配置文件"""
        if config_file is None:
            config_file = self.current_config_file
        
        file_path = Root / "Data" / config_file
        with open(file_path, "w", encoding="utf8") as f:
            f.write(json.dumps(self.content, ensure_ascii=False, indent=4))
    
    def load_config(self, config_file: str) -> bool:
        """載入指定的配置文件"""
        try:
            self.content = self.read(config_file)
            self.current_config_file = config_file
            return True
        except Exception as e:
            print(f"載入配置文件 {config_file} 失敗: {e}")
            return False
    
    def save_to_current_config(self) -> None:
        """保存到當前載入的配置文件"""
        self.save(self.current_config_file)
    
    def get_current_config_file(self) -> str:
        """獲取當前配置文件名稱"""
        return self.current_config_file

    def __getitem__(self, key: str) -> Any:
        return self.content.get(key, self.default.get(key))

    def __setitem__(self, key: str, value: Any) -> None:
        self.update(key, value)

    def __delitem__(self, key: str) -> None:
        self.delete(key)


Config = _Config()
