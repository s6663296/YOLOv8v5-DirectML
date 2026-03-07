import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import asyncio
from makcu import create_async_controller, MakcuConnectionError
import threading
import random
import string
import logging

# --- Custom Handler to bridge logging and Tkinter UI ---
class TkinterLogHandler(logging.Handler):
    """A logging handler that emits records to a Tkinter ScrolledText widget."""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        # This function needs to be thread-safe as it's called from the logger
        # The actual text insertion is scheduled to run in the main Tkinter thread
        self.text_widget.after(0, self.add_message, msg)

    def add_message(self, msg):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.configure(state='disabled')
        self.text_widget.yview(tk.END) # Auto-scroll to the bottom

class MakcuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Makcu 設備資訊 & 日誌")
        self.makcu_controller = None

        # --- UI Elements ---
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top frame for controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X)

        self.info_label = ttk.Label(top_frame, text="設備詳細資訊:")
        self.info_label.pack(anchor='w')
        self.info_text = tk.Text(top_frame, height=10, width=70)
        self.info_text.pack(fill=tk.X, expand=True, pady=(0, 5))

        self.version_label = ttk.Label(top_frame, text="韌體版本:")
        self.version_label.pack(anchor='w')
        self.version_value = tk.StringVar()
        self.version_entry = ttk.Entry(top_frame, textvariable=self.version_value, state='readonly')
        self.version_entry.pack(fill=tk.X, expand=True, pady=(0, 5))

        self.status_label = ttk.Label(top_frame, text="狀態: 未連線")
        self.status_label.pack(anchor='w', pady=(5, 10))

        # Button frame
        button_frame = ttk.Frame(top_frame)
        button_frame.pack(fill=tk.X)
        self.get_info_button = ttk.Button(button_frame, text="重新整理資訊", command=self.schedule_device_info, state=tk.DISABLED)
        self.get_info_button.pack(side=tk.LEFT, padx=(0, 5))
        self.get_version_button = ttk.Button(button_frame, text="取得韌體版本", command=self.schedule_firmware_version, state=tk.DISABLED)
        self.get_version_button.pack(side=tk.LEFT, padx=5)
        self.spoof_button = ttk.Button(button_frame, text="隨機序列欺騙", command=self.schedule_spoof_serial, state=tk.DISABLED)
        self.spoof_button.pack(side=tk.LEFT, padx=5)

        # --- NEW: Log Viewer ---
        log_frame = ttk.LabelFrame(main_frame, text="執行日誌")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log_widget = ScrolledText(log_frame, state='disabled', height=12, wrap=tk.WORD)
        self.log_widget.pack(fill=tk.BOTH, expand=True)

        # --- Setup Logging ---
        self.setup_logging()

        # --- Asyncio setup ---
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.thread.start()
        
        self.root.after(100, self.schedule_connection)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_logging(self):
        """Configure the logging module to output to the UI widget."""
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        
        # Create our custom handler
        ui_handler = TkinterLogHandler(self.log_widget)
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S')
        ui_handler.setFormatter(formatter)
        
        # Add the handler to the root logger
        log_root.addHandler(ui_handler)
        logging.info("日誌系統初始化完成。")

    def run_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def schedule_task(self, coro):
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def schedule_connection(self): self.schedule_task(self.connect_to_device())
    def schedule_device_info(self): self.schedule_task(self.get_device_info())
    def schedule_firmware_version(self): self.schedule_task(self.get_firmware_version())
    def schedule_spoof_serial(self): self.schedule_task(self.spoof_serial())

    async def connect_to_device(self):
        try:
            logging.info("正在嘗試連線至 Makcu 設備...")
            self.status_label.config(text="狀態: 正在連線...")
            self.makcu_controller = await create_async_controller()
            
            logging.info("控制器建立成功。")
            self.get_info_button.config(state=tk.NORMAL)
            self.get_version_button.config(state=tk.NORMAL)
            self.spoof_button.config(state=tk.NORMAL)
            
            await self.get_device_info()
            
        except MakcuConnectionError as e:
            logging.error(f"連線失敗，找不到設備: {e}", exc_info=False)
            self.status_label.config(text=f"狀態: 連線失敗 - {e}")
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"找不到設備，請確認已連接。\n詳細錯誤: {e}")
        except Exception as e:
            logging.error(f"連線時發生未預期的錯誤: {e}", exc_info=True)
            self.status_label.config(text=f"狀態: 發生未知錯誤")
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"發生預期外的錯誤: {e}")

    async def get_device_info(self):
        if not self.makcu_controller: return
        try:
            logging.info("正在取得設備詳細資訊...")
            info = await self.makcu_controller.get_device_info()
            logging.info(f"成功取得資訊: {info}")
            port = info.get('port', '未知連接埠')
            self.status_label.config(text=f"狀態: 已連線至 {port}")
            self.info_text.delete(1.0, tk.END)
            formatted_info = "\n".join([f"{key}: {value}" for key, value in info.items()])
            self.info_text.insert(tk.END, formatted_info)
        except Exception as e:
            logging.error(f"取得設備資訊時失敗: {e}", exc_info=True)
            self.status_label.config(text=f"狀態: 資訊取得失敗")

    async def get_firmware_version(self):
        if not self.makcu_controller: return
        try:
            logging.info("正在取得韌體版本...")
            version = await self.makcu_controller.get_firmware_version()
            logging.info(f"成功取得韌體版本: '{version}'")
            self.version_value.set(version)
        except Exception as e:
            logging.error(f"取得韌體版本時失敗: {e}", exc_info=True)
            self.version_value.set(f"發生錯誤: {e}")

    async def spoof_serial(self):
        if not self.makcu_controller: return
        try:
            random_serial = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
            logging.info(f"準備欺騙序列號。新序列號為: {random_serial}")
            self.status_label.config(text=f"狀態: 正在寫入序列號 {random_serial}...")
            
            logging.info("正在呼叫 controller.spoof_serial()...")
            await self.makcu_controller.spoof_serial(random_serial)
            logging.info("controller.spoof_serial() 呼叫完成，未拋出例外。")
            
            self.status_label.config(text="狀態: 序列號寫入完成！正在重新整理...")
            
            await asyncio.sleep(0.2) # 增加延遲，確保設備有時間處理寫入
            logging.info("正在重新整理設備資訊以驗證變更。")
            await self.get_device_info()
            
        except Exception as e:
            logging.error(f"序列號欺騙過程中發生錯誤: {e}", exc_info=True)
            self.status_label.config(text=f"狀態: 欺騙失敗 - {e}")

    def on_closing(self):
        logging.info("正在關閉應用程式...")
        if self.makcu_controller:
            future = asyncio.run_coroutine_threadsafe(self.makcu_controller.disconnect(), self.loop)
            try:
                future.result(timeout=2)
                logging.info("與設備連線已中斷。")
            except Exception as e:
                logging.warning(f"中斷連線時發生錯誤: {e}")
        
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MakcuApp(root)
    root.mainloop()
