"""VAD 段 → ASR 处理块的合并（纯逻辑，无重依赖，便于单测）。

从 asr_pipeline._merge_vad_segments 抽出，以便脱离 funasr/torch 直接单测合并规则。
"""


def merge_vad_segments(vad_segments, max_span_sec, max_gap_sec):
    """贪心合并相邻 VAD 段为 ASR 处理块。

    从首段起持续追加后续段，直到触发切组：
      (a) 合并后总跨度（首段 start → 末段 end）> max_span_sec，或
      (b) 与当前组的静音间隙（下一段 start − 当前组 end）> max_gap_sec。

    (b) 是关键约束：**绝不把大段静音并入同一 chunk**。合并后的 chunk 是一段连续音频切片
    （含内部静音），交给对齐器做词级时间戳；若 chunk 内含长静音，对齐器会把本应落在静音后
    语音段的词散布进静音区，产出「落进静音空档的幽灵词」——其时间戳比真实早约一个间隙，
    进而导致段级 start/end 回退乱序。限制被桥接的静音长度即从源头杜绝此问题。

    max_gap_sec 允许桥接的短停顿（同一话语被 VAD 拆成的碎片），远小于会触发误对齐的长静音。

    参数以秒计；vad_segments 为按时间升序的 [(start_ms, end_ms), ...]，返回同形合并结果。
    """
    if not vad_segments:
        return []

    max_span_ms = int(max_span_sec * 1000)
    max_gap_ms = int(max_gap_sec * 1000)
    merged = []
    group_start, group_end = vad_segments[0]

    for start_ms, end_ms in vad_segments[1:]:
        within_span = end_ms - group_start <= max_span_ms
        within_gap = start_ms - group_end <= max_gap_ms
        if within_span and within_gap:
            group_end = end_ms
        else:
            merged.append((group_start, group_end))
            group_start, group_end = start_ms, end_ms

    merged.append((group_start, group_end))
    return merged


def vad_voiced_duration_sec(start_sec, end_sec, vad_segments):
    """句子区间 [start_sec, end_sec]（秒）内的 VAD 语音总时长（秒）。

    对每个 VAD 段与句子区间求交并累加。vad_segments 为按 start 升序、互不重叠的
    [(start_ms, end_ms), ...]（VADEngine.detect 输出）。由于各交集都被钳在句子区间内且
    VAD 段互不重叠，累加结果**恒 ≤ 句子跨度（end - start）**——这是调用方做合理性自检
    （vad 总时长不得大于句子总时长）的数学保证。

    空区间或 start >= end 返回 0.0。
    """
    lo = float(start_sec) * 1000.0
    hi = float(end_sec) * 1000.0
    if hi <= lo:
        return 0.0
    total_ms = 0.0
    for s_ms, e_ms in vad_segments:
        if e_ms <= lo:
            continue          # 该段完全在句子左侧
        if s_ms >= hi:
            break             # 已升序，后续段都在句子右侧
        total_ms += min(e_ms, hi) - max(s_ms, lo)
    return total_ms / 1000.0
