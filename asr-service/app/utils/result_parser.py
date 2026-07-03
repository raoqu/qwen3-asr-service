"""qwen_asr transcribe 结果解析共享工具（离线管线与实时会话共用）。

从 ASRTranscription 结果中提取纯文本与单词级时间戳，
统一 asr_pipeline 与 stream_session 两处的解析逻辑，避免拷贝发散。
"""


def extract_text(results) -> str:
    """从 qwen_asr transcribe 结果中提取纯文本"""
    if not results:
        return ""
    if isinstance(results, str):
        return results
    if isinstance(results, list):
        texts = []
        for item in results:
            if hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, dict):
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return "".join(texts)
    if hasattr(results, "text"):
        return results.text
    return str(results)


def extract_words(results, offset_sec: float) -> list[dict] | None:
    """从 ASR 结果中提取单词级时间戳（带偏移修正）。

    兼容两种结果形态：
    - transformers/vLLM 后端：ASRTranscription 对象，
      .time_stamps -> ForcedAlignResult.items -> [ForcedAlignItem(text,start_time,end_time)]
    - MLX 后端：dict，预解析为 {"text", "words": [{"text","start","end"}, ...]}
      （start/end 为相对 chunk 起点的秒，此处统一加 offset_sec 修正为绝对秒）
    """
    if not results or not isinstance(results, list):
        return None

    words = []
    for item in results:
        if isinstance(item, dict):
            for w in item.get("words") or []:
                words.append({
                    "text": w["text"],
                    "start": round(w["start"] + offset_sec, 3),
                    "end": round(w["end"] + offset_sec, 3),
                })
            continue
        # ASRTranscription.time_stamps -> ForcedAlignResult.items -> [ForcedAlignItem]
        ts = getattr(item, "time_stamps", None)
        if ts is None:
            continue
        for w in getattr(ts, "items", []):
            words.append({
                "text": w.text,
                "start": round(w.start_time + offset_sec, 3),
                "end": round(w.end_time + offset_sec, 3),
            })
    return words if words else None


# ─── 词级时间戳对齐校验（sanitize）────────────────────────────────────
# ForcedAligner 在部分 chunk 上会失效：输出相对时间超出音频长度、词序错乱、或把整段
# 文本塌缩进极短时间窗（如 28 词压进 3.3s、54 词压进 0.3s）。坏词时间戳流入分句会
# 产生段级时间戳漂移/回退乱序。整组校验不过 → 丢弃该 chunk 全部词，回退 chunk 级
# 时间戳（准确、只是粒度粗）；文本不受影响。

_WORD_BOUND_TOL_SEC = 0.5      # 词允许超出 chunk 边界的容差（秒）：超出→对齐器越界，整组拒收
_WORD_ORDER_TOL_SEC = 0.5      # 词间允许的时间回退容差（秒）：更大回退→词序错乱，整组拒收
_COLLAPSE_MAX_RATE = 12.0      # 词速硬上限（词/秒）：任何语言正常语速的数倍，超出→塌缩
_COLLAPSE_COVERAGE = 0.25      # 塌缩判定的跨度占比下限：词跨度 < chunk 时长的此比例
_COLLAPSE_MIN_WORDS = 15       # 塌缩判定的最少词数（避免误伤"短句+长静音"的正常 chunk）
_COLLAPSE_SOFT_RATE = 8.0      # 低覆盖时的词速软上限（词/秒）：与覆盖率联合判定


def sanitize_words(words, offset_sec: float, duration_sec: float):
    """校验一个 chunk 的词级时间戳是否为有效对齐，返回 (words | None, reason | None)。

    校验不过整组丢弃（返回 None + 原因）——半好半坏的词序列无法可靠区分，宁可回退
    chunk 级时间戳。轻微越界（<= 容差）钳回边界而不拒收。
    """
    if not words:
        return None, None
    lo = float(offset_sec)
    hi = lo + float(duration_sec)

    prev_start = None
    for w in words:
        ws, we = float(w["start"]), float(w["end"])
        if ws < lo - _WORD_BOUND_TOL_SEC or we > hi + _WORD_BOUND_TOL_SEC:
            return None, (f"词时间越界: {w['text']!r} {ws:.2f}-{we:.2f} 超出 "
                          f"chunk [{lo:.2f}, {hi:.2f}] 容差 {_WORD_BOUND_TOL_SEC}s")
        if prev_start is not None and ws < prev_start - _WORD_ORDER_TOL_SEC:
            return None, f"词序错乱: {w['text']!r}@{ws:.2f} 回退到 {prev_start:.2f} 之前"
        prev_start = ws

    if len(words) >= _COLLAPSE_MIN_WORDS:
        span = max(float(w["end"]) for w in words) - min(float(w["start"]) for w in words)
        rate = len(words) / max(span, 0.1)
        coverage = span / max(float(duration_sec), 0.1)
        if rate > _COLLAPSE_MAX_RATE:
            return None, f"对齐塌缩: {len(words)} 词压进 {span:.2f}s（{rate:.1f} 词/秒）"
        if coverage < _COLLAPSE_COVERAGE and rate > _COLLAPSE_SOFT_RATE:
            return None, (f"对齐塌缩: {len(words)} 词仅覆盖 chunk 的 "
                          f"{coverage:.0%}（{rate:.1f} 词/秒）")

    # 轻微越界钳回边界（不改变通过判定）
    out = []
    for w in words:
        ws = min(max(float(w["start"]), lo), hi)
        we = min(max(float(w["end"]), ws), hi)
        out.append({"text": w["text"], "start": round(ws, 3), "end": round(we, 3)})
    return out, None
