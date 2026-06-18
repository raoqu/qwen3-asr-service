"""
mlx_asr_engine.py
─────────────────────────────────────────────────────────────────────
MLX ASR 引擎（Apple Silicon）：基于 mlx-qwen3-asr 的 Qwen3-ASR，Metal GPU 推理。

与 OpenVINOASREngine / QwenASREngine 保持相同接口，供 ASRPipeline 与流式会话使用。
纯 MLX，无 torch / openvino / funasr 依赖。模型权重从 HuggingFace 拉取并本地缓存。

设计要点（见 docs/mig_plan.md §〇·五 实测）：
- 实测 ASR 较 OpenVINO CPU 提速 3.7–4.2×（RTF 0.18 → ~0.044），CER 1–4%。
- 说话人分离仍走现有 CAM++ 路径，本引擎仅做 ASR（始终 diarize=False，
  不触碰 mlx-qwen3-asr 内置的 pyannote 分离，避免其首次 HF 下载挂起）。
- 与 OpenVINO 引擎一致：不产出词级时间戳（align_enabled=False），按 chunk 逐段识别。
"""
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)

# model_size → HuggingFace 仓库
_MODEL_REPO = {
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
}

# pipeline 传入的语言代码 → mlx-qwen3-asr 期望的语言名称（与 OpenVINO 引擎对齐）
_LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
}


class MLXASREngine:
    """MLX (Metal GPU) ASR 引擎，Qwen3-ASR via mlx-qwen3-asr。"""

    BACKEND = "mlx"

    def __init__(self, model_size: str = "0.6b", dtype: str = "float16"):
        self._model_size = model_size
        self._dtype = dtype
        self._model = None
        self._config = None
        self._M = None
        # MLX 的 Stream/计算上下文是线程局部的，且 mlx-qwen3-asr 在加载时即捕获 GPU
        # stream——模型必须在「加载它的那个线程」上推理，否则报
        # "There is no Stream(gpu, N) in current thread"。
        # 因此用一个专用单线程 executor：load 与所有 transcribe 都在它上面执行，
        # 既保证线程亲和性，又顺带把离线管线/流式会话的并发推理串行化（max_workers=1）。
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-asr")

    def load(self):
        """加载 MLX 模型（首次会从 HuggingFace 下载权重）。在专用线程上执行。"""
        self._executor.submit(self._load_impl).result()

    def _load_impl(self):
        try:
            import mlx.core as mx
            import mlx_qwen3_asr as M
        except ImportError:
            raise ImportError(
                "MLX 模式需要 mlx 与 mlx-qwen3-asr（仅 Apple Silicon），"
                "请执行: pip install mlx mlx-qwen3-asr"
            )
        if self._model_size not in _MODEL_REPO:
            raise ValueError(f"不支持的 model_size: {self._model_size}")

        self._M = M
        dtype_map = {
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
        }
        dt = dtype_map.get(self._dtype, mx.float16)
        repo = _MODEL_REPO[self._model_size]

        logger.info(f"开始加载 MLX ASR 模型（Metal）: {repo} dtype={self._dtype} ...")
        self._model, self._config = M.load_model(repo, dtype=dt)
        logger.info(f"MLX ASR 模型已加载: size={self._model_size}, device=mlx/metal")

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> list[dict]:
        """对单个音频文件执行 ASR 识别。返回 [{"text": str}]（与 OpenVINO 引擎一致）。"""
        if self._model is None:
            raise RuntimeError("ASR 模型未加载，请先调用 load()")

        lang = self._map_language(language)

        def _run():
            res = self._M.transcribe(
                audio_path, model=self._model, language=lang, diarize=False)
            return [{"text": (res.text or "").strip()}]

        return self._executor.submit(_run).result()

    def transcribe_array(
        self,
        audio,
        sr: int = 16000,
        language: str | None = None,
    ) -> list[dict]:
        """对内存音频数组执行 ASR 识别（实时流式会话用，不落盘）。

        参数:
            audio: np.ndarray（float32，单声道，16kHz）
            sr: 采样率
            language: 语言（可选）
        返回:
            [{"text": str}]
        """
        if self._model is None:
            raise RuntimeError("ASR 模型未加载，请先调用 load()")

        audio = np.asarray(audio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        lang = self._map_language(language)

        def _run():
            # mlx-qwen3-asr 接受 (array, sr) 元组，内部重采样到 16k 单声道
            res = self._M.transcribe(
                (audio, sr), model=self._model, language=lang, diarize=False)
            return [{"text": (res.text or "").strip()}]

        return self._executor.submit(_run).result()

    def _map_language(self, language: str | None) -> str | None:
        """短语言代码 → mlx 期望的语言全称；已是全称或未知则原样透传。"""
        if language is None:
            return None
        return _LANG_MAP.get(language, language)

    def unload(self):
        """释放资源。在专用线程上清理模型，再关停 executor。"""
        def _clear():
            self._model = None
            self._config = None
        try:
            self._executor.submit(_clear).result()
        finally:
            self._executor.shutdown(wait=True)
        logger.info("MLX ASR 模型已卸载")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def align_enabled(self) -> bool:
        # 与 OpenVINO 引擎一致：当前不产出词级时间戳（如需可后续接入 mlx forced_aligner）
        return False
