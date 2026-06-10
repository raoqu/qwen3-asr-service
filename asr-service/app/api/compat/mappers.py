"""兼容层结果映射（纯函数，单测主战场）。

数据源：pipeline result dict（asr_pipeline.py:209）——`segments[].start/end` = **秒**(float)，
`segments[].words[].start/end` = 秒。OpenAI verbose_json 全用秒，直取；srt/vtt 由秒格式化为
时间轴。DashScope（Phase 2，毫秒）与实时（Phase 3）映射后续并入本模块。
"""
from __future__ import annotations


def _fmt_timestamp(seconds: float | None, *, sep: str = ",") -> str:
    """秒 → HH:MM:SS<sep>mmm（srt 用 ','，vtt 用 '.'）；None/负值钳为 0。"""
    if seconds is None or seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000))
    h, ms_total = divmod(ms_total, 3_600_000)
    m, ms_total = divmod(ms_total, 60_000)
    s, ms = divmod(ms_total, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _collect_words(segments: list[dict]) -> list[dict]:
    """汇总各 segment 词级时间戳到顶层 words[]（text→word，start/end 秒直取）。"""
    words = []
    for seg in segments:
        for w in seg.get("words") or []:
            words.append({"word": w.get("text", ""),
                          "start": w.get("start"), "end": w.get("end")})
    return words


def _duration(segments: list[dict]) -> float:
    """全段时长 = max(end)；空段为 0。"""
    return max((seg.get("end", 0.0) or 0.0) for seg in segments) if segments else 0.0


def _verbose_segment(idx: int, seg: dict) -> dict:
    """单段 → OpenAI verbose_json segment。本服务无来源的字段填占位（reference §3.3 注明）。"""
    return {
        "id": idx,
        "seek": 0,
        "start": seg.get("start", 0.0),
        "end": seg.get("end", 0.0),
        "text": seg.get("text", ""),
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": 0.0,
        "compression_ratio": 0.0,
        "no_speech_prob": 0.0,
    }


def result_to_openai(result: dict, *, response_format: str,
                     want_word_ts: bool, language: str | None):
    """pipeline result → OpenAI 响应。

    json/verbose_json → dict；text/srt/vtt → str（调用方用 PlainTextResponse 包装）。
    """
    segments = result.get("segments") or []
    full_text = result.get("full_text", "")

    if response_format == "json":
        return {"text": full_text}
    if response_format == "text":
        return full_text
    if response_format == "srt":
        return result_to_srt(segments)
    if response_format == "vtt":
        return result_to_vtt(segments)

    # verbose_json
    out = {
        "task": "transcribe",
        "language": language or result.get("language"),
        "duration": _duration(segments),
        "text": full_text,
        "segments": [_verbose_segment(i, seg) for i, seg in enumerate(segments)],
    }
    if want_word_ts:
        words = _collect_words(segments)
        if words:
            out["words"] = words
    return out


def result_to_srt(segments: list[dict]) -> str:
    """segments → SRT 字幕（时间轴 HH:MM:SS,mmm，序号从 1）。"""
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _fmt_timestamp(seg.get("start", 0.0), sep=",")
        end = _fmt_timestamp(seg.get("end", 0.0), sep=",")
        lines += [str(i), f"{start} --> {end}", seg.get("text", ""), ""]
    return "\n".join(lines)


def result_to_vtt(segments: list[dict]) -> str:
    """segments → WebVTT 字幕（时间轴 HH:MM:SS.mmm）。"""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _fmt_timestamp(seg.get("start", 0.0), sep=".")
        end = _fmt_timestamp(seg.get("end", 0.0), sep=".")
        lines += [f"{start} --> {end}", seg.get("text", ""), ""]
    return "\n".join(lines)
