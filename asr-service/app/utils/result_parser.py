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
