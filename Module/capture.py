# Module/capture.py
"""
管理螢幕截圖並提供畫面進行推論。
支援 MSS 螢幕截圖、DXGI 高效截圖和 OBS 虛擬相機三種擷取模式。
"""
import threading
import time
from collections import deque
from typing import Any
import mss
import numpy as np
import cv2
from Module.logger import logger

class ScreenCaptureManager:
    """管理螢幕截圖並提供畫面進行推論。支援 MSS、DXGI 和 OBS 虛擬相機三種模式。"""
    
    def __init__(self, screen_bbox: dict, exit_event: threading.Event, new_frame_event: threading.Event, target_fps: int = 240, capture_source: str = "mss", obs_ip: str = "", obs_port: int = 1234, capture_monitor_index: int = 1, capture_monitor_bounds: dict | None = None):
        """初始化螢幕截圖管理器。
        
        Args:
            screen_bbox: 截圖區域的邊界框
            exit_event: 退出事件
            new_frame_event: 新幀事件
            target_fps: 目標幀率
            capture_source: 截圖來源，"mss"、"dxgi" 或 "obs"
            obs_ip: OBS UDP 串流 IP 地址（空字串表示使用本地虛擬相機）
            obs_port: OBS UDP 串流端口
            capture_monitor_index: 擷取顯示器索引（1-based）
            capture_monitor_bounds: 擷取顯示器邊界資訊（left/top/width/height）
        """
        self.screen_bbox = screen_bbox
        self.exit_event = exit_event
        self.new_frame_event = new_frame_event
        self.target_fps = target_fps
        self.capture_source = capture_source
        self.obs_ip = obs_ip
        self.obs_port = obs_port
        try:
            self.capture_monitor_index = max(1, int(capture_monitor_index))
        except (TypeError, ValueError):
            self.capture_monitor_index = 1
        self.capture_monitor_bounds = capture_monitor_bounds or {}
        
        self.frame_lock = threading.Lock()
        self.frame_queue = deque(maxlen=1)  # 只保留最新一幀以減少延遲
        self.latest_frame = None
        self.latest_frame_timestamp = None  # 記錄幀的擷取時間戳
        self.capture_fps = 0
        self.restart_required = threading.Event()
        self.thread = None
        
        # OBS 虛擬相機相關
        self._video_capture = None
        
        # DXGI 相關 (bettercam)
        self._bettercam_camera = None
        
    def start(self):
        """啟動螢幕截圖執行緒。"""
        if self.capture_source == "obs":
            # 使用 UDP 串流監聽模式接收來自遠端 OBS 的畫面
            self.thread = threading.Thread(target=self._capture_thread_obs_udp, daemon=True, name="capture_thread_obs_udp")
            if self.obs_ip:
                logger.info(f"使用 OBS UDP 串流接收模式 (監聽 IP: {self.obs_ip}, Port: {self.obs_port})")
            else:
                logger.info(f"使用 OBS UDP 串流接收模式 (監聯所有介面, Port: {self.obs_port})")
        elif self.capture_source == "dxgi":
            self.thread = threading.Thread(target=self._capture_thread_dxgi, daemon=True, name="capture_thread_dxgi")
            logger.info("使用 DXGI 高效截圖模式")
        else:
            self.thread = threading.Thread(target=self._capture_thread_mss, daemon=True, name="capture_thread_mss")
            logger.info("使用 MSS 螢幕截圖模式")
        self.thread.start()
        logger.info("螢幕截圖執行緒已啟動")
        
    def _capture_thread_mss(self):
        """MSS 螢幕截圖執行緒函式。"""
        try:
            with mss.mss() as sct:
                frame_times = deque(maxlen=100)
                last_fps_display = time.time()
                
                while not self.exit_event.is_set() and not self.restart_required.is_set():
                    try:
                        start_time = time.time()
                          
                        # 記錄截圖時間戳（使用高精度時間）
                        capture_timestamp = time.perf_counter()
                        frame_raw = sct.grab(self.screen_bbox)
                        
                        # 效能優化：使用 .raw (BGRA) 而非 .rgb，避免 mss 內部的 BGRA→RGB 轉換
                        # 直接取前 3 通道 (BGR) 並建立獨立副本
                        bgra_shape = (frame_raw.height, frame_raw.width, 4)
                        frame_bgr = np.array(
                            np.frombuffer(frame_raw.raw, dtype=np.uint8).reshape(bgra_shape)[:, :, :3]
                        )
                        
                        with self.frame_lock:
                            self.latest_frame = frame_bgr
                            self.latest_frame_timestamp = capture_timestamp  # 儲存擷取時間戳
                            self.frame_queue.append((frame_bgr, capture_timestamp))  # 幀與時間戳一起存儲
                            
                        self.new_frame_event.set()
                        
                        frame_time = time.time() - start_time
                        frame_times.append(frame_time)
                        
                        if time.time() - last_fps_display > 2.0:
                            if frame_times:
                                avg_frame_time = sum(frame_times) / len(frame_times)
                                self.capture_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                                logger.debug(f"螢幕截圖 FPS: {self.capture_fps:.1f}")
                            last_fps_display = time.time()
                            
                        target_frame_time = 1.0 / self.target_fps
                        sleep_time = max(0, target_frame_time - frame_time)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        
                    except Exception as e:
                        logger.error(f"螢幕截圖錯誤: {e}")
                        time.sleep(0.5)
                        
        except Exception as e:
            logger.error(f"致命的螢幕截圖錯誤: {e}")
            self.restart_required.set()

    def _capture_thread_dxgi(self):
        """DXGI Desktop Duplication 截圖執行緒函式（使用 bettercam）。
        
        終極效能與精準控制版：
        1. 使用 Windows 高精度等候計時器 (WaitableTimer) 取代 Python 不精準的 time.sleep() (誤差往往達 15ms)。
           利用作業系統原生的中斷機制，以極精準的頻率 (如 120 FPS 或 240 FPS) 喚醒擷取迴圈，在等候期間 CPU 消耗為 0%。
        2. 全無鎖單執行緒 (Lock-Free Async)：不啟動 bettercam 背景執行緒，自己負責同步 grab() 並立即丟給 AI 推論。
        3. 自帶 Zero-Copy Video Mode：如果 DXGI 偵測到畫面 (Monitor) 完全沒有任何更新，會回傳 None，
           我們直接重複使用上一張實體圖片記憶體的指標，100% 零 CPU 損耗達成高頻率補幀送圖。
        4. 推論連續記憶體優化：以 .copy() 保證從 1080p 切割出 320x320 的陣列是 Contiguous Array。
        """
        try:
            import bettercam
            from bettercam.util.timer import create_high_resolution_timer, set_periodic_timer, wait_for_timer, cancel_timer, INFINITE, WAIT_FAILED
            
            output_idx = max(0, self.capture_monitor_index - 1)
            use_primary_fallback = False

            try:
                # 採用 BGRA 取得最乾淨未經處理的原始 Buffer
                camera = bettercam.create(output_idx=output_idx, output_color="BGRA")
            except Exception as e:
                logger.warning(f"指定 DXGI 顯示器 output_idx={output_idx} 失敗，將回退到主顯示器: {e}")
                camera = bettercam.create(output_idx=0, output_color="BGRA")
                output_idx = 0
                use_primary_fallback = True

            if camera is None:
                logger.error("bettercam 初始化失敗，無法建立擷取器")
                self.restart_required.set()
                return
            self._bettercam_camera = camera

            output_width = int(camera.width)
            output_height = int(camera.height)

            target_width = int(self.screen_bbox["width"])
            target_height = int(self.screen_bbox["height"])
            if target_width > output_width or target_height > output_height:
                logger.warning(
                    f"DXGI 目標區域 {target_width}x{target_height} 大於輸出解析度 {output_width}x{output_height}，將自動縮小"
                )
                target_width = min(target_width, output_width)
                target_height = min(target_height, output_height)

            if use_primary_fallback:
                local_left = max(0, (output_width - target_width) // 2)
                local_top = max(0, (output_height - target_height) // 2)
            else:
                monitor_left = int(self.capture_monitor_bounds.get("left", 0))
                monitor_top = int(self.capture_monitor_bounds.get("top", 0))
                local_left = int(self.screen_bbox["left"]) - monitor_left
                local_top = int(self.screen_bbox["top"]) - monitor_top
                local_left = max(0, min(local_left, output_width - target_width))
                local_top = max(0, min(local_top, output_height - target_height))

            region = (
                local_left,
                local_top,
                local_left + target_width,
                local_top + target_height,
            )

            logger.info(
                f"DXGI 擷取設定: output_idx={output_idx}, region=({region[0]},{region[1]},{target_width},{target_height})"
            )
            
            logger.info("bettercam 已初始化，啟動配備高精度硬體計時器的終極無鎖擷取模式")
            
            frame_times = deque(maxlen=100)
            last_fps_display = time.time()
            capture_fps = max(120, int(self.target_fps))
            
            # --- Windows 高精度計時器設定 ---
            timer_handle = None
            try:
                if capture_fps > 0:
                    period_ms = max(1, 1000 // capture_fps)
                    timer_handle = create_high_resolution_timer()
                    set_periodic_timer(timer_handle, period_ms)
                    logger.info(f"高精度計時器設定成功，週期為 {period_ms} 毫秒")
            except Exception as e:
                logger.error(f"高精度計時器設定失敗，將依靠系統迴圈速度: {e}")
                timer_handle = None
            
            while not self.exit_event.is_set() and not self.restart_required.is_set():
                # 在迴圈頂部利用硬體計時器阻塞 (0% CPU 消耗)
                if timer_handle:
                    res = wait_for_timer(timer_handle, INFINITE)
                    if res == WAIT_FAILED:
                        logger.error("高精度計時器等待失敗")
                        continue

                try:
                    start_time = time.perf_counter()
                    
                    # 進行無鎖直接呼叫，完全不影響任何其他線程，不用搶鎖
                    frame_bgra = camera.grab(region=region)
                    
                    if frame_bgra is None or frame_bgra.size == 0:
                        # 模擬 Video_Mode 機制，但由於不透過 buffer 不拷貝，完全是 Zero-Copy
                        if self.latest_frame is not None:
                            frame_bgr = self.latest_frame
                        else:
                            time.sleep(0.001)  # 首幀還沒進來前短暫退讓
                            continue
                    else:
                        # 有新畫面的「局部更新」才做 BGR 轉換與連續記憶體排列優化
                        if frame_bgra.shape[2] == 4:
                            frame_bgr = frame_bgra[:, :, :3].copy()
                        else:
                            frame_bgr = frame_bgra

                    capture_timestamp = time.perf_counter()
                    
                    with self.frame_lock:
                        self.latest_frame = frame_bgr
                        self.latest_frame_timestamp = capture_timestamp
                        self.frame_queue.append((frame_bgr, capture_timestamp))
                    
                    self.new_frame_event.set()
                    
                    # 計算推論與供圖耗時
                    frame_time = time.perf_counter() - start_time
                    frame_times.append(frame_time)
                    
                    if time.time() - last_fps_display > 2.0:
                        if frame_times:
                            avg_frame_time = sum(frame_times) / len(frame_times)
                            actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                            # Add information about frame drop vs idle
                            is_idle = frame_bgra is None or frame_bgra.size == 0
                            logger.debug(f"DXGI 截圖供檔 FPS: {actual_fps:.1f} | 狀態: {'靜止補幀' if is_idle else '實體抓取'} | 耗時: {avg_frame_time*1000:.2f}ms")
                        last_fps_display = time.time()
                        
                except Exception as e:
                    logger.error(f"DXGI 截圖錯誤: {e}")
                    time.sleep(0.5)
                    
        except ImportError:
            logger.error("bettercam 套件未安裝，無法使用 DXGI 截圖模式。請執行: pip install bettercam")
            self.restart_required.set()
        except Exception as e:
            logger.error(f"致命的 DXGI 截圖錯誤: {e}")
            self.restart_required.set()
        finally:
            if 'timer_handle' in locals() and timer_handle:
                try:
                    from bettercam.util.timer import cancel_timer
                    cancel_timer(timer_handle)
                    logger.info("已釋放高精度硬體計時器資源")
                except Exception as e:
                    logger.error(f"釋放高精度計時器失敗: {e}")

            if self._bettercam_camera is not None:
                try:
                    self._bettercam_camera.stop()
                except Exception:
                    pass
                try:
                    del self._bettercam_camera
                except Exception:
                    pass
                self._bettercam_camera = None
                logger.info("DXGI 擷取器已釋放 (bettercam)")
    
    def _capture_thread_obs_udp(self):
        """OBS UDP 串流截圖執行緒函式。使用 OpenCV + 獨立讀取執行緒，永遠只保留最新幀。"""
        import os
        
        # 設定 FFmpeg 超低延遲環境變數（OpenCV 內建使用 FFmpeg）
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'probesize;32|'
            'analyzeduration;0|'
            'fflags;nobuffer+discardcorrupt|'
            'flags;low_delay|'
            'framedrop;1|'
            'max_delay;0'
        )
        
        # 目標解析度
        target_width = self.screen_bbox["width"]
        target_height = self.screen_bbox["height"]
        
        # 建立 UDP 監聽地址
        if self.obs_ip:
            udp_url = f"udp://{self.obs_ip}:{self.obs_port}?overrun_nonfatal=1&fifo_size=50000"
        else:
            udp_url = f"udp://@:{self.obs_port}?overrun_nonfatal=1&fifo_size=50000"
        
        logger.info(f"正在監聽 OBS UDP 串流 (超低延遲模式): Port {self.obs_port}")
        
        # 共享變數：最新幀
        latest_frame_data: list[Any] = [None, None]  # [frame_bgr, timestamp]
        latest_frame_lock = threading.Lock()
        reader_running = [True]
        reader_error: list[str | None] = [None]
        
        def video_reader(video_cap, target_w, target_h):
            """獨立執行緒：持續從 VideoCapture 讀取，只保留最新幀"""
            while reader_running[0]:
                try:
                    ret, frame_bgr = video_cap.read()
                    if ret and frame_bgr is not None:
                        timestamp = time.perf_counter()
                        # 縮放到目標大小
                        if frame_bgr.shape[1] != target_w or frame_bgr.shape[0] != target_h:
                            frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        # 覆蓋舊幀，只保留最新
                        with latest_frame_lock:
                            latest_frame_data[0] = frame_bgr
                            latest_frame_data[1] = timestamp
                    else:
                        # 讀取失敗
                        time.sleep(0.001)
                except Exception as e:
                    reader_error[0] = str(e)
                    break
        
        try:
            # 開啟 VideoCapture
            self._video_capture = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
            
            if not self._video_capture.isOpened():
                logger.error(f"無法開啟 UDP 監聽: {udp_url}")
                self.restart_required.set()
                return
            
            # 禁用 OpenCV 內部緩衝
            self._video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            
            logger.info("OBS UDP 串流已連接")
            
            # 啟動讀取執行緒
            reader_thread = threading.Thread(
                target=video_reader,
                args=(self._video_capture, target_width, target_height),
                daemon=True,
                name="opencv_video_reader"
            )
            reader_thread.start()
            logger.info("獨立讀取執行緒已啟動 - 永遠只保留最新幀")
            
            frame_times = deque(maxlen=100)
            last_fps_display = time.time()
            last_frame_id = None  # 用來檢測是否是新幀
            
            while not self.exit_event.is_set() and not self.restart_required.is_set():
                try:
                    start_time = time.time()
                    
                    # 檢查讀取執行緒是否有錯誤
                    if reader_error[0] is not None:
                        logger.error(f"讀取執行緒錯誤: {reader_error[0]}")
                        break
                    
                    # 從共享變數獲取最新幀（非阻塞）
                    frame_bgr = None
                    timestamp = None
                    with latest_frame_lock:
                        if latest_frame_data[0] is not None:
                            frame_bgr = latest_frame_data[0]
                            timestamp = latest_frame_data[1]
                            latest_frame_data[0] = None  # 取走後清空
                    
                    if frame_bgr is None:
                        # 還沒有新幀，短暫等待
                        time.sleep(0.0005)  # 0.5ms
                        continue
                    
                    # 檢查是否為重複幀（使用 id）
                    frame_id = id(frame_bgr)
                    if frame_id == last_frame_id:
                        time.sleep(0.0005)
                        continue
                    last_frame_id = frame_id
                    
                    # 轉換 BGR 到 RGB
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    with self.frame_lock:
                        self.latest_frame = frame_rgb
                        self.latest_frame_timestamp = timestamp
                        self.frame_queue.append((frame_rgb, timestamp))
                    
                    self.new_frame_event.set()
                    
                    frame_time = time.time() - start_time
                    frame_times.append(frame_time)
                    
                    if time.time() - last_fps_display > 2.0:
                        if frame_times:
                            avg_frame_time = sum(frame_times) / len(frame_times)
                            self.capture_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                            logger.debug(f"OBS UDP 串流 FPS: {self.capture_fps:.1f}")
                        last_fps_display = time.time()
                    
                except Exception as e:
                    logger.error(f"OBS UDP 串流擷取錯誤: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"致命的 OBS UDP 串流錯誤: {e}")
            self.restart_required.set()
        finally:
            reader_running[0] = False
            if self._video_capture is not None:
                self._video_capture.release()
                self._video_capture = None
            
    def get_frame(self):
        """獲取最新的畫面。
        
        Returns:
            tuple: (frame, timestamp) 或 (None, None)
                - frame: numpy array 的畫面資料
                - timestamp: 畫面擷取時的高精度時間戳
        """
        with self.frame_lock:
            if not self.frame_queue:
                return None, None
            # 返回最新的幀和時間戳
            frame, timestamp = self.frame_queue[-1]
            return frame, timestamp
            
    def stop(self):
        """發送訊號停止執行緒。"""
        self.exit_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
        if self._bettercam_camera is not None:
            try:
                self._bettercam_camera.stop()
            except Exception:
                pass
            try:
                del self._bettercam_camera
            except Exception:
                pass
            self._bettercam_camera = None
        logger.info("螢幕截圖已停止。")

