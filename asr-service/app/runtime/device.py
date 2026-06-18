import logging
import platform

logger = logging.getLogger(__name__)


def is_apple_silicon() -> bool:
    """是否运行在 Apple Silicon（macOS arm64）。"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    """MLX ASR 后端是否可用（Apple Silicon + 已装 mlx / mlx-qwen3-asr）。"""
    if not is_apple_silicon():
        return False
    import importlib.util
    return (importlib.util.find_spec("mlx") is not None
            and importlib.util.find_spec("mlx_qwen3_asr") is not None)


def resolve_asr_backend(requested: str, is_cpu: bool) -> str:
    """确定 ASR 推理后端。

    参数:
        requested: "auto" | "openvino" | "mlx" | "qwen"
        is_cpu: resolve_device() 结果是否为 cpu（即无 CUDA）

    返回:
        "mlx" | "openvino" | "qwen"

    auto 策略：CUDA → qwen；CPU 且 Apple Silicon 且 MLX 可用 → mlx；否则 → openvino。
    """
    if requested == "mlx":
        if not mlx_available():
            raise RuntimeError(
                "请求 MLX 后端，但当前非 Apple Silicon 或未安装 mlx / mlx-qwen3-asr"
            )
        return "mlx"
    if requested == "openvino":
        return "openvino"
    if requested == "qwen":
        return "qwen"
    # auto
    if not is_cpu:
        return "qwen"
    if mlx_available():
        return "mlx"
    return "openvino"


def detect_device() -> dict:
    """
    检测运行设备，返回设备信息。

    返回:
        {"type": "cuda"|"cpu", "vram_gb": float|None, "name": str|None}
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            name = torch.cuda.get_device_name(0)
            logger.info(f"检测到 GPU: {name}, VRAM: {vram:.1f}GB")
            return {"type": "cuda", "vram_gb": round(vram, 1), "name": name}
    except ImportError:
        pass

    logger.info("使用 CPU 模式")
    return {"type": "cpu", "vram_gb": None, "name": None}


def resolve_device(requested: str, device_info: dict | None = None) -> str:
    """
    根据用户请求和硬件情况，确定最终使用的设备。

    参数:
        requested: "auto" | "cuda" | "cpu"
        device_info: detect_device() 的返回值，避免重复检测

    返回:
        "cuda" | "cpu"
    """
    info = device_info or detect_device()

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if info["type"] != "cuda":
            raise RuntimeError("请求使用 CUDA 但未检测到可用 GPU")
        return "cuda"

    # auto 模式
    return info["type"]


def auto_select_model_size(vram_gb: float | None) -> str:
    """根据显存自动选择模型大小"""
    if vram_gb is None:
        return "0.6b"
    if vram_gb >= 6:
        return "1.7b"
    return "0.6b"


def should_disable_align(device: str, vram_gb: float | None) -> bool:
    """判断是否需要强制关闭对齐"""
    if device == "cpu":
        return True
    if vram_gb is not None and vram_gb < 4:
        return True
    return False
