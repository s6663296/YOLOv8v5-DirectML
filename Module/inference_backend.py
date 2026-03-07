"""推理後端相關常數與工具函式。"""

from typing import Dict, List

DEFAULT_BACKEND = "dml"

BACKEND_LABELS: Dict[str, str] = {
    "dml": "DirectML (AMD/Intel/NVIDIA)",
    "cuda": "CUDA (NVIDIA)",
    "cpu": "CPU (通用)",
    "trt": "TensorRT (NVIDIA 高效能)",
}

ONNX_PROVIDERS: Dict[str, List[str]] = {
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


def normalize_backend(backend: str) -> str:
    value = (backend or "").strip().lower()
    if value in BACKEND_LABELS:
        return value
    return DEFAULT_BACKEND


def backend_to_ui_text(backend: str) -> str:
    normalized = normalize_backend(backend)
    return BACKEND_LABELS.get(normalized, BACKEND_LABELS[DEFAULT_BACKEND])


def backend_from_ui_text(text: str) -> str:
    ui_text = (text or "").strip()
    for backend, label in BACKEND_LABELS.items():
        if ui_text == label:
            return backend

    upper_text = ui_text.upper()
    if "TENSORRT" in upper_text:
        return "trt"
    if "CUDA" in upper_text:
        return "cuda"
    if "CPU" in upper_text:
        return "cpu"
    return DEFAULT_BACKEND


def get_onnx_providers(backend: str) -> List[str]:
    effective_backend = get_effective_onnx_backend(backend)
    return ONNX_PROVIDERS[effective_backend]


def get_effective_onnx_backend(backend: str) -> str:
    normalized = normalize_backend(backend)
    if normalized in ONNX_PROVIDERS:
        return normalized
    return DEFAULT_BACKEND


def should_use_tensorrt(backend: str, model_path: str) -> bool:
    normalized = normalize_backend(backend)
    path = str(model_path or "")
    return normalized == "trt" or path.lower().endswith(".engine")
