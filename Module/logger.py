import datetime
import logging
import os
import threading
import time
from io import StringIO
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING  # noqa: F401

from colorama import Fore, Style, init

from Module.config import Config, Root


def _parse_level(level_name: str, default: int) -> int:
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(str(level_name).upper(), default)


def _parse_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_log_level() -> int:
    return _parse_level(Config.get("log_level", "INFO"), logging.INFO)


def get_terminal_log_level() -> int:
    return _parse_level(Config.get("terminal_log_level", "INFO"), logging.INFO)


class TerminalNoiseFilter(logging.Filter):
    """Reduce terminal noise while preserving essential runtime signals."""

    NOISY_INFO_PREFIXES = (
        "YOLO 置信度更改為:",
        "IOU 閾值更改為:",
        "模型輸入尺寸更改為:",
        "為螢幕大小更新了 AimLogic 參數:",
        "Aim logic parameters updated from saved settings.",
        "壓槍觸發按鍵設定為:",
        "PID Kp 設定為:",
        "PID Ki 設定為:",
        "PID Kd 設定為:",
        "FPS 已啟用。",
        "FPS 已停用。",
        "延遲覆蓋層已啟用。",
        "延遲覆蓋層已停用。",
        "FPS 浮水印 ",
        "預覽功能已 ",
        "UI顯示，預覽更新計時器已重新啟動",
        "UI隱藏，預覽更新計時器已停止。",
        "預覽視窗已重新建立。",
        "預覽視窗已關閉並釋放資源。",
        "最後一幀預覽影像已釋放。",
        "Thread pool periodic stats:",
        "Thread pool mouse stats:",
        "Frame manager stats:",
        "目標類別選單已建立",
        "偵測目標類別更改為:",
        "截圖速率已更新為:",
        "預覽與繪圖更新率已同步至:",
        "程序優先權已設定為:",
        "模型輸入名稱:",
        "模型輸出名稱:",
        "模型類別名稱已載入:",
        "正在預熱 ONNX 模型...",
        "正在使用原生 YOLO",
        "YOLOv5 類別數量:",
        "YOLOv8 類別數量:",
        "YOLOv11 類別數量:",
        "YOLOv12 類別數量:",
        "AsyncInferencePipeline: 推論迴圈開始",
        "AsyncInferencePipeline: 推論迴圈結束",
        "AsyncInferencePipeline: 推論執行緒已啟動",
        "AsyncInferencePipeline: 已停止",
    )
    KEEP_INFO_KEYWORDS = (
        "推理後端",
        "provider",
    )

    def __init__(self, compact_mode: bool = True, dedupe_seconds: float = 3.0):
        super().__init__()
        self.compact_mode = compact_mode
        self.dedupe_seconds = max(0.0, dedupe_seconds)
        self._last_seen = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True

        if record.levelno < logging.INFO:
            return False

        message = record.getMessage()

        if any(keyword in message for keyword in self.KEEP_INFO_KEYWORDS):
            return True

        if self.compact_mode and message.startswith(self.NOISY_INFO_PREFIXES):
            return False

        if self.dedupe_seconds > 0 and self._is_duplicate(message):
            return False

        return True

    def _is_duplicate(self, message: str) -> bool:
        now = time.monotonic()
        previous = self._last_seen.get(message)
        self._last_seen[message] = now

        if len(self._last_seen) > 512:
            threshold = now - (self.dedupe_seconds * 4)
            self._last_seen = {
                key: timestamp
                for key, timestamp in self._last_seen.items()
                if timestamp >= threshold
            }

        return previous is not None and (now - previous) < self.dedupe_seconds


class CustomFormatter(logging.Formatter):
    def format(self, record):
        color = self._get_color(record.levelname)
        record.color = color
        return super().format(record)

    def _get_color(self, level_name):
        colors = {
            "DEBUG": Fore.CYAN,
            "INFO": Fore.BLUE,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Style.BRIGHT,
        }
        return colors.get(level_name, Fore.WHITE)


class _Logger:
    def __init__(self, log_file_prefix=Root / "logs"):
        init(autoreset=True)

        console_log_level_int = get_terminal_log_level()
        self.file_log_level_int = INFO
        compact_mode = _parse_bool(Config.get("terminal_compact_logs", True), True)
        dedupe_seconds = _parse_float(Config.get("terminal_dedupe_seconds", 3.0), 3.0)

        os.makedirs(log_file_prefix, exist_ok=True)
        self.log_file_prefix = log_file_prefix
        self.logger = logging.getLogger("Custom Logger")
        self.logger.setLevel(get_log_level())

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level_int)
        console_handler.addFilter(
            TerminalNoiseFilter(
                compact_mode=compact_mode,
                dedupe_seconds=dedupe_seconds,
            )
        )

        colored_formatter = CustomFormatter(
            f"{Fore.GREEN}{Style.BRIGHT}%(asctime)s{Style.RESET_ALL} "
            f"%(color)s[%(levelname)s]{Style.RESET_ALL} "
            f"{Fore.WHITE}%(message)s{Style.RESET_ALL}",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(colored_formatter)
        self.logger.addHandler(console_handler)

        self.file_handler = None
        self.current_log_date = None
        self.lock = threading.Lock()
        self.log_stream = StringIO()

    def _ensure_log_file_created(self):
        today = datetime.datetime.now().date()
        if self.file_handler is None or today != self.current_log_date:
            with self.lock:
                if self.file_handler is not None and today != self.current_log_date:
                    self.logger.removeHandler(self.file_handler)
                    self.file_handler.close()

                self.current_log_date = today
                self.log_file = os.path.join(self.log_file_prefix, f"{today}.log")
                self.file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
                self.file_handler.setLevel(self.file_log_level_int)
                self.file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                self.logger.addHandler(self.file_handler)

    def _format_message(self, *args):
        return " ".join(str(arg) for arg in args)

    def debug(self, *args) -> None:
        self._ensure_log_file_created()
        with self.lock:
            self.logger.debug(self._format_message(*args))

    def info(self, *args) -> None:
        self._ensure_log_file_created()
        with self.lock:
            self.logger.info(self._format_message(*args))

    def warning(self, *args) -> None:
        self._ensure_log_file_created()
        with self.lock:
            self.logger.warning(self._format_message(*args))

    def warn(self, *args) -> None:
        self.warning(*args)

    def error(self, *args) -> None:
        self._ensure_log_file_created()
        with self.lock:
            self.logger.error(self._format_message(*args))

    def critical(self, *args) -> None:
        self._ensure_log_file_created()
        with self.lock:
            self.logger.critical(self._format_message(*args))

    def fatal(self, *args) -> None:
        self.critical(*args)

    def _generate_log_output(self):
        while True:
            if log_content := self.log_stream.getvalue():
                self.log_stream.seek(0)
                self.log_stream.truncate(0)
                yield log_content
            else:
                yield ""


logger = _Logger()
