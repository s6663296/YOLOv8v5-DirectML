#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TensorRT 模型轉換 GUI：整合 ONNX 與 PT 轉換流程。"""

from __future__ import annotations

import contextlib
import io
import os
import queue
import shutil
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def ensure_tensorrt_available() -> tuple[bool, Any]:
    """確認 TensorRT Python 綁定可用。"""
    try:
        import tensorrt as trt

        return True, trt
    except ImportError:
        print("錯誤: TensorRT 未安裝。請先安裝 TensorRT。")
        print("安裝步驟:")
        print("  pip install \"C:\\TensorRT-10.13.3.9\\python\\tensorrt-10.13.3.9-cp310-none-win_amd64.whl\"")
        return False, None


def ensure_ultralytics_available() -> tuple[bool, Any]:
    """確認 ultralytics 可用。"""
    try:
        from ultralytics import YOLO

        return True, YOLO
    except ImportError:
        print("錯誤: ultralytics 未安裝。請執行: pip install ultralytics")
        return False, None


def convert_onnx_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    use_fp16: bool = True,
    workspace_size_gb: float = 4.0,
    verbose: bool = False,
) -> bool:
    """將 ONNX 模型轉換為 TensorRT Engine。"""
    trt_ok, trt = ensure_tensorrt_available()
    if not trt_ok:
        return False

    logger_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(logger_level)

    print(f"\n{'=' * 60}")
    print("ONNX -> TensorRT 模型轉換工具")
    print(f"{'=' * 60}")
    print(f"TensorRT 版本: {trt.__version__}")
    print(f"輸入模型: {onnx_path}")
    print(f"輸出 Engine: {engine_path}")
    print(f"精度模式: {'FP16' if use_fp16 else 'FP32'}")
    print(f"工作空間: {workspace_size_gb} GB")
    print(f"{'=' * 60}\n")

    if not os.path.exists(onnx_path):
        print(f"錯誤: 找不到 ONNX 檔案: {onnx_path}")
        return False

    builder = trt.Builder(logger)
    if builder is None:
        print("錯誤: 無法創建 TensorRT Builder")
        return False

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    if network is None:
        print("錯誤: 無法創建 TensorRT Network")
        return False

    parser = trt.OnnxParser(network, logger)
    if parser is None:
        print("錯誤: 無法創建 ONNX Parser")
        return False

    print("正在解析 ONNX 模型...")
    start_time = time.time()
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    success = parser.parse(onnx_data)
    if not success:
        print("錯誤: ONNX 解析失敗")
        for i in range(parser.num_errors):
            print(f"  - {parser.get_error(i)}")
        return False

    parse_time = time.time() - start_time
    print(f"ONNX 解析完成，耗時: {parse_time:.2f} 秒")

    print("\n網路資訊:")
    print(f"  輸入數量: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    [{i}] {inp.name}: {inp.shape}")
    print(f"  輸出數量: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    [{i}] {out.name}: {out.shape}")
    print(f"  層數量: {network.num_layers}")

    config = builder.create_builder_config()
    if config is None:
        print("錯誤: 無法創建 Builder Config")
        return False

    workspace_bytes = int(workspace_size_gb * 1024 * 1024 * 1024)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("\n已啟用 FP16 精度模式")
        else:
            print("\n警告: 此 GPU 不支援快速 FP16，將使用 FP32")

    has_dynamic_shape = False
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        for dim in inp.shape:
            if dim == -1:
                has_dynamic_shape = True
                break

    if has_dynamic_shape:
        print("\n偵測到動態形狀，設定優化配置...")
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            min_shape = (1, 3, 320, 320)
            opt_shape = (1, 3, 640, 640)
            max_shape = (1, 3, 1280, 1280)
            profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
            print(f"  {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
        config.add_optimization_profile(profile)

    print("\n正在構建 TensorRT Engine (這可能需要幾分鐘)...")
    build_start = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("錯誤: 構建 Engine 失敗")
        return False

    build_time = time.time() - build_start
    print(f"Engine 構建完成，耗時: {build_time:.2f} 秒")

    print(f"\n正在保存 Engine 到: {engine_path}")
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_size = os.path.getsize(engine_path)
    print(f"Engine 大小: {engine_size / 1024 / 1024:.2f} MB")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"轉換成功！總耗時: {total_time:.2f} 秒")
    print(f"輸出檔案: {engine_path}")
    print(f"{'=' * 60}\n")
    return True


def convert_pt_to_engine(
    pt_path: str,
    engine_path: str,
    imgsz: int = 640,
    use_fp16: bool = True,
    workspace_size_gb: float = 4.0,
    verbose: bool = False,
) -> bool:
    """將 PyTorch 模型直接轉換為 TensorRT Engine。"""
    yolo_ok, YOLO = ensure_ultralytics_available()
    if not yolo_ok:
        return False

    trt_ok, trt = ensure_tensorrt_available()
    if not trt_ok:
        return False

    print(f"\n{'=' * 60}")
    print("PyTorch -> TensorRT 模型轉換工具")
    print(f"{'=' * 60}")
    print(f"TensorRT 版本: {trt.__version__}")
    print(f"輸入模型: {pt_path}")
    print(f"輸出 Engine: {engine_path}")
    print(f"輸入尺寸: {imgsz}x{imgsz}")
    print(f"精度模式: {'FP16' if use_fp16 else 'FP32'}")
    print(f"工作空間: {workspace_size_gb} GB")
    print(f"{'=' * 60}\n")

    if not os.path.exists(pt_path):
        print(f"錯誤: 找不到 PyTorch 檔案: {pt_path}")
        return False

    try:
        start_time = time.time()

        print("步驟 1/3: 載入 PyTorch 模型...")
        model = YOLO(pt_path)
        print(f"  模型載入完成: {model.model.yaml.get('yaml_file', 'unknown')}")

        print("\n步驟 2/3: 導出為 TensorRT Engine...")
        print("  (這可能需要幾分鐘，請耐心等待...)")

        output_dir = os.path.dirname(os.path.abspath(engine_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        export_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=use_fp16,
            workspace=workspace_size_gb,
            verbose=verbose,
        )

        if export_path and os.path.exists(export_path):
            if os.path.abspath(export_path) != os.path.abspath(engine_path):
                shutil.move(export_path, engine_path)
                print(f"  Engine 已移動到: {engine_path}")

        if not os.path.exists(engine_path):
            print("錯誤: TensorRT Engine 生成失敗")
            return False

        engine_size = os.path.getsize(engine_path)
        total_time = time.time() - start_time

        print("\n步驟 3/3: 驗證輸出...")
        print(f"  Engine 大小: {engine_size / 1024 / 1024:.2f} MB")

        print(f"\n{'=' * 60}")
        print(f"轉換成功！總耗時: {total_time:.2f} 秒")
        print(f"輸出檔案: {engine_path}")
        print(f"{'=' * 60}\n")

        return True

    except Exception as e:
        print(f"\n錯誤: 轉換失敗 - {e}")
        if verbose:
            traceback.print_exc()
        return False


def convert_pt_to_onnx_to_engine(
    pt_path: str,
    engine_path: str,
    imgsz: int = 640,
    use_fp16: bool = True,
    workspace_size_gb: float = 4.0,
    verbose: bool = False,
) -> bool:
    """先將 PT 轉 ONNX，再轉 TensorRT Engine。"""
    yolo_ok, YOLO = ensure_ultralytics_available()
    if not yolo_ok:
        return False

    trt_ok, _trt = ensure_tensorrt_available()
    if not trt_ok:
        return False

    print(f"\n{'=' * 60}")
    print("PyTorch -> ONNX -> TensorRT 模型轉換")
    print(f"{'=' * 60}")

    try:
        start_time = time.time()

        print("\n步驟 1/4: 載入 PyTorch 模型...")
        model = YOLO(pt_path)

        print("\n步驟 2/4: 導出為 ONNX...")
        onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)

        if not onnx_path or not os.path.exists(onnx_path):
            print("錯誤: ONNX 導出失敗")
            return False

        print(f"  ONNX 導出成功: {onnx_path}")

        print("\n步驟 3/4: 轉換為 TensorRT Engine...")
        success = convert_onnx_to_tensorrt(
            onnx_path=onnx_path,
            engine_path=engine_path,
            use_fp16=use_fp16,
            workspace_size_gb=workspace_size_gb,
            verbose=verbose,
        )

        if success:
            total_time = time.time() - start_time
            print(f"\n{'=' * 60}")
            print(f"轉換成功！總耗時: {total_time:.2f} 秒")
            print(f"{'=' * 60}\n")

        return success

    except Exception as e:
        print(f"\n錯誤: 轉換失敗 - {e}")
        if verbose:
            traceback.print_exc()
        return False


@dataclass
class ConversionTask:
    """單次轉換任務描述。"""

    mode: str
    input_path: str
    output_path: str
    use_fp16: bool = True
    workspace_gb: float = 4.0
    verbose: bool = False
    pt_size: int = 640
    pt_method: str = "direct"


class ConversionService:
    """統一管理不同來源模型的 TensorRT 轉換流程。"""

    MODE_ONNX = "onnx"
    MODE_PT = "pt"

    @staticmethod
    def validate_task(task: ConversionTask) -> tuple[bool, str]:
        if task.mode not in (ConversionService.MODE_ONNX, ConversionService.MODE_PT):
            return False, "不支援的轉換模式。"

        if not task.input_path:
            return False, "請選擇輸入模型。"
        if not os.path.exists(task.input_path):
            return False, f"找不到輸入檔案: {task.input_path}"

        if task.mode == ConversionService.MODE_ONNX and not task.input_path.lower().endswith(".onnx"):
            return False, "ONNX 模式僅接受 .onnx 檔案。"
        if task.mode == ConversionService.MODE_PT and not task.input_path.lower().endswith(".pt"):
            return False, "PyTorch 模式僅接受 .pt 檔案。"

        if not task.output_path:
            return False, "請指定輸出檔案。"
        if not task.output_path.lower().endswith(".engine"):
            return False, "輸出檔案必須是 .engine。"

        if task.workspace_gb <= 0:
            return False, "Workspace 必須大於 0。"

        if task.mode == ConversionService.MODE_PT:
            if task.pt_size <= 0:
                return False, "輸入尺寸必須大於 0。"
            if task.pt_method not in ("direct", "onnx"):
                return False, "PT 轉換方法必須是 direct 或 onnx。"

        return True, ""

    @staticmethod
    def suggest_engine_path(input_path: str) -> str:
        if not input_path:
            return ""
        base, _ = os.path.splitext(input_path)
        return f"{base}.engine"

    @staticmethod
    def run(task: ConversionTask) -> bool:
        if task.mode == ConversionService.MODE_ONNX:
            return convert_onnx_to_tensorrt(
                onnx_path=task.input_path,
                engine_path=task.output_path,
                use_fp16=task.use_fp16,
                workspace_size_gb=task.workspace_gb,
                verbose=task.verbose,
            )

        if task.pt_method == "onnx":
            return convert_pt_to_onnx_to_engine(
                pt_path=task.input_path,
                engine_path=task.output_path,
                imgsz=task.pt_size,
                use_fp16=task.use_fp16,
                workspace_size_gb=task.workspace_gb,
                verbose=task.verbose,
            )

        return convert_pt_to_engine(
            pt_path=task.input_path,
            engine_path=task.output_path,
            imgsz=task.pt_size,
            use_fp16=task.use_fp16,
            workspace_size_gb=task.workspace_gb,
            verbose=task.verbose,
        )


class QueueWriter(io.TextIOBase):
    """把 print 輸出導向 queue，供 GUI log 區顯示。"""

    def __init__(self, out_queue: queue.Queue):
        super().__init__()
        self._queue = out_queue

    def write(self, s: str) -> int:
        if s:
            self._queue.put(s)
        return len(s)

    def flush(self) -> None:
        return None


class TensorRTConverterGUI(tk.Tk):
    """主視窗。"""

    MODE_LABELS = {
        "ONNX -> TensorRT (.engine)": ConversionService.MODE_ONNX,
        "PyTorch (.pt) -> TensorRT (.engine)": ConversionService.MODE_PT,
    }

    PT_METHOD_LABELS = {
        "direct (ultralytics 直接輸出)": "direct",
        "onnx (先轉 ONNX 再轉 TRT)": "onnx",
    }

    def __init__(self) -> None:
        super().__init__()
        self.title("TensorRT 模型轉換工具")
        self.geometry("980x680")
        self.minsize(880, 600)

        self.log_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None

        self.mode_var = tk.StringVar(value="ONNX -> TensorRT (.engine)")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.precision_var = tk.StringVar(value="fp16")
        self.workspace_var = tk.StringVar(value="4")
        self.verbose_var = tk.BooleanVar(value=False)
        self.pt_size_var = tk.StringVar(value="640")
        self.pt_method_var = tk.StringVar(value="direct (ultralytics 直接輸出)")
        self.status_var = tk.StringVar(value="就緒")

        self._build_ui()
        self._on_mode_changed()
        self.after(120, self._poll_log_queue)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(8, weight=1)

        ttk.Label(container, text="轉換模式:").grid(row=0, column=0, sticky="w", pady=4)
        mode_combo = ttk.Combobox(
            container,
            textvariable=self.mode_var,
            state="readonly",
            values=list(self.MODE_LABELS.keys()),
        )
        mode_combo.grid(row=0, column=1, sticky="ew", pady=4)
        mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_changed())

        ttk.Label(container, text="輸入模型:").grid(row=1, column=0, sticky="w", pady=4)
        self.input_entry = ttk.Entry(container, textvariable=self.input_var)
        self.input_entry.grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(container, text="瀏覽", command=self._browse_input).grid(row=1, column=2, padx=(8, 0), pady=4)

        ttk.Label(container, text="輸出 Engine:").grid(row=2, column=0, sticky="w", pady=4)
        self.output_entry = ttk.Entry(container, textvariable=self.output_var)
        self.output_entry.grid(row=2, column=1, sticky="ew", pady=4)

        output_btns = ttk.Frame(container)
        output_btns.grid(row=2, column=2, padx=(8, 0), pady=4, sticky="e")
        ttk.Button(output_btns, text="自動", command=self._fill_output_from_input).pack(side=tk.LEFT)
        ttk.Button(output_btns, text="另存", command=self._browse_output).pack(side=tk.LEFT, padx=(6, 0))

        options = ttk.LabelFrame(container, text="通用選項", padding=10)
        options.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        for col in range(6):
            options.columnconfigure(col, weight=1)

        ttk.Label(options, text="精度:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(options, text="FP16 (推薦)", value="fp16", variable=self.precision_var).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(options, text="FP32", value="fp32", variable=self.precision_var).grid(row=0, column=2, sticky="w")

        ttk.Label(options, text="Workspace (GB):").grid(row=0, column=3, sticky="e")
        ttk.Entry(options, textvariable=self.workspace_var, width=8).grid(row=0, column=4, sticky="w")
        ttk.Checkbutton(options, text="Verbose 日誌", variable=self.verbose_var).grid(row=0, column=5, sticky="w")

        self.pt_options = ttk.LabelFrame(container, text="PyTorch 模式選項", padding=10)
        self.pt_options.grid(row=4, column=0, columnspan=3, sticky="ew", pady=4)
        for col in range(4):
            self.pt_options.columnconfigure(col, weight=1)

        ttk.Label(self.pt_options, text="輸入尺寸:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.pt_options, textvariable=self.pt_size_var, width=8).grid(row=0, column=1, sticky="w")

        ttk.Label(self.pt_options, text="轉換方法:").grid(row=0, column=2, sticky="e")
        ttk.Combobox(
            self.pt_options,
            textvariable=self.pt_method_var,
            state="readonly",
            values=list(self.PT_METHOD_LABELS.keys()),
        ).grid(row=0, column=3, sticky="ew")

        action_row = ttk.Frame(container)
        action_row.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        action_row.columnconfigure(0, weight=1)

        self.start_button = ttk.Button(action_row, text="開始轉換", command=self._start_conversion)
        self.start_button.pack(side=tk.LEFT)
        ttk.Button(action_row, text="清空日誌", command=self._clear_log).pack(side=tk.LEFT, padx=(8, 0))

        self.status_label = ttk.Label(action_row, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT)

        log_frame = ttk.LabelFrame(container, text="轉換日誌", padding=6)
        log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=16)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _get_mode(self) -> str:
        return self.MODE_LABELS.get(self.mode_var.get(), ConversionService.MODE_ONNX)

    def _on_mode_changed(self) -> None:
        mode = self._get_mode()
        if mode == ConversionService.MODE_PT:
            self.pt_options.grid()
        else:
            self.pt_options.grid_remove()

        if self.input_var.get().strip():
            self.output_var.set(ConversionService.suggest_engine_path(self.input_var.get().strip()))

    def _browse_input(self) -> None:
        mode = self._get_mode()
        if mode == ConversionService.MODE_ONNX:
            file_types = [("ONNX 模型", "*.onnx"), ("所有檔案", "*.*")]
        else:
            file_types = [("PyTorch 模型", "*.pt"), ("所有檔案", "*.*")]

        path = filedialog.askopenfilename(title="選擇輸入模型", filetypes=file_types)
        if path:
            self.input_var.set(path)
            self.output_var.set(ConversionService.suggest_engine_path(path))

    def _browse_output(self) -> None:
        input_path = self.input_var.get().strip()
        initial_name = os.path.basename(ConversionService.suggest_engine_path(input_path)) if input_path else "model.engine"
        path = filedialog.asksaveasfilename(
            title="選擇輸出 Engine 路徑",
            defaultextension=".engine",
            filetypes=[("TensorRT Engine", "*.engine")],
            initialfile=initial_name,
        )
        if path:
            if not path.lower().endswith(".engine"):
                path += ".engine"
            self.output_var.set(path)

    def _fill_output_from_input(self) -> None:
        input_path = self.input_var.get().strip()
        if not input_path:
            messagebox.showwarning("提示", "請先選擇輸入模型。")
            return
        self.output_var.set(ConversionService.suggest_engine_path(input_path))

    def _build_task(self) -> tuple[ConversionTask | None, str]:
        mode = self._get_mode()
        input_path = self.input_var.get().strip()
        output_path = self.output_var.get().strip()

        try:
            workspace_gb = float(self.workspace_var.get().strip())
        except ValueError:
            return None, "Workspace 必須是數字。"

        pt_size = 640
        if mode == ConversionService.MODE_PT:
            try:
                pt_size = int(self.pt_size_var.get().strip())
            except ValueError:
                return None, "輸入尺寸必須是整數。"

        task = ConversionTask(
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            use_fp16=self.precision_var.get() == "fp16",
            workspace_gb=workspace_gb,
            verbose=self.verbose_var.get(),
            pt_size=pt_size,
            pt_method=self.PT_METHOD_LABELS.get(self.pt_method_var.get(), "direct"),
        )

        ok, err = ConversionService.validate_task(task)
        if not ok:
            return None, err
        return task, ""

    def _start_conversion(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return

        task, err = self._build_task()
        if not task:
            messagebox.showerror("參數錯誤", err)
            return

        self._append_log("\n" + "=" * 64 + "\n")
        self._append_log(f"模式: {self.mode_var.get()}\n")
        self._append_log(f"輸入: {task.input_path}\n")
        self._append_log(f"輸出: {task.output_path}\n")
        self._append_log("=" * 64 + "\n")

        self.start_button.configure(state=tk.DISABLED)
        self.status_var.set("轉換中...")

        self.worker_thread = threading.Thread(target=self._run_task, args=(task,), daemon=True)
        self.worker_thread.start()

    def _run_task(self, task: ConversionTask) -> None:
        writer = QueueWriter(self.log_queue)
        success = False
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                success = ConversionService.run(task)
        except Exception:
            self.log_queue.put(traceback.format_exc())
            success = False
        finally:
            self.log_queue.put(("__DONE__", success))

    def _poll_log_queue(self) -> None:
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(item, tuple) and item and item[0] == "__DONE__":
                self._on_task_done(bool(item[1]))
            else:
                self._append_log(str(item))

        self.after(120, self._poll_log_queue)

    def _on_task_done(self, success: bool) -> None:
        self.start_button.configure(state=tk.NORMAL)
        if success:
            self.status_var.set("完成")
            self._append_log("\n[完成] 轉換成功。\n")
            messagebox.showinfo("完成", "轉換成功！")
        else:
            self.status_var.set("失敗")
            self._append_log("\n[失敗] 轉換失敗，請檢查日誌。\n")
            messagebox.showerror("失敗", "轉換失敗，請查看日誌內容。")

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", tk.END)


def main() -> None:
    app = TensorRTConverterGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
