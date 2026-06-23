"""正则后处理分句（--regex）。

当 cfg.REGEX_SEGMENT 开启且对齐器启用（segment 带词级时间戳 words）时，对默认
segment_sentences 的输出再做一遍「标点正则 + 时长约束」重切，替换其结果；未开启时
管线行为与现状完全一致（本模块不被调用）。

四参数两阶段（默认值见 config.py / 暴露见 arg_schema）：

  阶段 1 —— 标点正则切句（第一优先级）：
    以 .。?？!！ 为句末边界（保护小数 3.14 / 连续点 ... / .env / 单字母缩写 e.g.，
    复用 sentence_segmenter 的英文句点判定）。标点边界仅当左侧「已累积句长」
    >= short_sec 才真正切；短于 short_sec 继续向后累积（避免切出过短碎句）。
    切完后任何仍 > long_sec 的句子按词间停顿/时间「硬切」到 <= long_sec（达上限强制切）。

  阶段 2 —— VAD 时长精修：
    a) 长于 vad_max_sec 的句子：按内部「最大词间停顿」处递归切开（哪怕未超 long_sec），
       仅在存在明显停顿（>= _MIN_SPLIT_GAP）时切，找不到干净停顿则保留。
    b) 短于 vad_min_sec 的句子：并入相邻句（优先并入前句；合并后不得超过 long_sec）。

不跨说话人：相邻 segment 说话人不同、或 [识别失败] 标记块，均作为硬边界分块处理，
块内拼接 words（绝对时间戳）后整体重切。说话人最终标签由管线 4.7 步按新句界重算。

concat 不变量：输出各句文本顺序拼接 == 输入文本顺序拼接（块内切/并只重排边界，不增删字符）。
"""
from app.pipeline.sentence_segmenter import (
    _word_positions, _spans, _pieces, _is_english_period_end,
)

_FAIL_MARK = "[识别失败]"
# 句末标点正则集合（用户指定）：中文 。？！ + 英文 ? ！(全角) ! ；英文 . 走保护判定。
_REGEX_END_NOPROTECT = "。？！?！!"
_MIN_SPLIT_GAP = 0.15   # 阶段 2a 视为「可切停顿」的最小词间隙（秒），低于此不在停顿处切


def regex_segment(segments, *, long_sec, short_sec, vad_max_sec, vad_min_sec):
    """对已带 words 的句子级 segments 做正则后处理重切，返回同形 segments。

    无 words 或 [识别失败] 的 segment 作为硬边界原样保留；说话人切换处不跨块合并。
    """
    if not segments:
        return []
    out = []
    block = []
    prev_spk = None

    def flush():
        nonlocal block
        if block:
            out.extend(_segment_block(block, long_sec, short_sec, vad_max_sec, vad_min_sec))
            block = []

    for seg in segments:
        is_fail = (seg.get("text") or "").strip() == _FAIL_MARK
        spk = seg.get("speaker")
        if is_fail or not seg.get("words"):
            flush()
            out.append(seg)          # 硬边界：原样保留
            prev_spk = None
            continue
        if block and spk != prev_spk:
            flush()
        block.append(seg)
        prev_spk = spk
    flush()
    return out


def _segment_block(block, long_sec, short_sec, vad_max_sec, vad_min_sec):
    """对单个「同说话人、连续、带 words」的块整体重切。"""
    full_text = "".join(s["text"] for s in block)
    words = []
    for s in block:
        words.extend(s.get("words") or [])
    speaker = block[0].get("speaker")
    if not full_text.strip() or not words:
        return list(block)

    positions = _word_positions(full_text, words)
    n = len(full_text)
    cuts = sorted(i + 1 for i in range(n) if _is_regex_end(full_text, i) and 0 < i + 1 < n)
    spans = _spans(0, n, cuts)
    pieces = _pieces(full_text, words, positions,
                     float(block[0]["start"]), float(block[-1]["end"]), spans, speaker)

    sentences = _accumulate(pieces, short_sec)                      # 阶段 1：标点 + short
    sentences = _flatten(_split_by_pause(s, long_sec, hard=True)    # 阶段 1：long 强制切
                         for s in sentences)
    sentences = _flatten(_split_by_pause(s, vad_max_sec, hard=False)  # 阶段 2a：VAD 上限切
                         for s in sentences)
    sentences = _merge_short(sentences, vad_min_sec, long_sec)        # 阶段 2b：VAD 下限并

    return [_finalize(s) for s in sentences if (s.get("text") or "").strip()]


# ─── 阶段 1：标点累积 ─────────────────────────────────────────────────

def _accumulate(pieces, short_sec):
    """按标点单元累积为句：左侧累积句长 < short_sec（或单元无词）则继续并入，否则在此切。"""
    sentences = []
    cur = None
    for p in pieces:
        if cur is None:
            cur = _copy(p)
            continue
        if not p.get("words") or _dur(cur) < short_sec:
            _merge_into(cur, p)
        else:
            sentences.append(cur)
            cur = _copy(p)
    if cur is not None:
        sentences.append(cur)
    return sentences


# ─── 时长切分（停顿优先，硬模式按时间兜底）────────────────────────────

def _split_by_pause(s, max_sec, *, hard):
    """把时长 > max_sec 的句子切到 <= max_sec。

    优先在最大词间停顿（>= _MIN_SPLIT_GAP）处切；找不到干净停顿时：hard=True 按时间
    兜底硬切（达上限强制切），hard=False 保留不切（无明显停顿不强行切完整句）。
    """
    words = s.get("words")
    if _dur(s) <= max_sec or not words or len(words) < 2:
        return [s]
    gi, gmax = _largest_gap(words)
    if gmax >= _MIN_SPLIT_GAP:
        left, right = _split_at_word(s, gi + 1)
        return _split_by_pause(left, max_sec, hard=hard) + _split_by_pause(right, max_sec, hard=hard)
    if hard:
        return _hard_split(s, max_sec)
    return [s]


def _hard_split(s, max_sec):
    """无明显停顿的超长句：从句首起按累计时长达到 max_sec 处切，递归切尾。"""
    words = s["words"]
    t0 = words[0]["start"]
    k = len(words)
    for idx in range(1, len(words)):
        if words[idx]["start"] - t0 > max_sec:
            k = idx
            break
    if k >= len(words):
        return [s]
    left, right = _split_at_word(s, k)
    return [left] + _hard_split(right, max_sec)


def _split_at_word(s, k):
    """在第 k 个词之前把句子切成两句（1 <= k < len(words)）；时间取词级时间戳。"""
    text, words = s["text"], s["words"]
    cut = _word_positions(text, words)[k]
    lw, rw = words[:k], words[k:]
    left = {"text": text[:cut], "words": lw or None,
            "start": lw[0]["start"], "end": lw[-1]["end"], "speaker": s.get("speaker")}
    right = {"text": text[cut:], "words": rw or None,
             "start": rw[0]["start"], "end": rw[-1]["end"], "speaker": s.get("speaker")}
    return left, right


def _largest_gap(words):
    """返回最大词间停顿的 (前词下标, 间隙秒)。"""
    gi, gmax = 0, -1.0
    for k in range(len(words) - 1):
        gap = words[k + 1]["start"] - words[k]["end"]
        if gap > gmax:
            gi, gmax = k, gap
    return gi, gmax


# ─── 阶段 2b：过短句合并 ──────────────────────────────────────────────

def _merge_short(sentences, vad_min_sec, long_sec):
    """短于 vad_min_sec 的句子并入相邻句：优先并入前句，合并后不得超过 long_sec。"""
    out = []
    for s in sentences:
        s = _copy(s)
        if out and _dur(s) < vad_min_sec and (s["end"] - out[-1]["start"]) <= long_sec:
            _merge_into(out[-1], s)                   # 并入前句
            continue
        out.append(s)
    # 首句过短且无前句可并：尝试并入其后句（不超 long_sec）
    if len(out) >= 2 and _dur(out[0]) < vad_min_sec \
            and (out[1]["end"] - out[0]["start"]) <= long_sec:
        _merge_into_front(out[1], out[0])
        out.pop(0)
    return out


# ─── 句结构小工具 ─────────────────────────────────────────────────────

def _dur(s):
    return float(s["end"]) - float(s["start"])


def _copy(p):
    return {"text": p["text"], "words": list(p["words"]) if p.get("words") else None,
            "start": p["start"], "end": p["end"], "speaker": p.get("speaker")}


def _merge_into(a, b):
    """把 b 追加到 a 之后；仅 b 有词时扩展 end（避免无词碎片的估时污染边界）。"""
    a["text"] += b["text"]
    if b.get("words"):
        a["words"] = (a.get("words") or []) + list(b["words"])
        a["end"] = max(a["end"], b["end"])


def _merge_into_front(b, a):
    """把 a 前置到 b 之前（用于首句过短并入后句）。"""
    b["text"] = a["text"] + b["text"]
    if a.get("words"):
        b["words"] = list(a["words"]) + (b.get("words") or [])
        b["start"] = min(a["start"], b["start"])


def _finalize(s):
    seg = {"start": round(float(s["start"]), 3),
           "end": round(float(max(s["end"], s["start"])), 3),
           "text": s["text"]}
    if s.get("words"):
        seg["words"] = s["words"]
    if s.get("speaker") is not None:
        seg["speaker"] = s["speaker"]
    return seg


def _flatten(iterable_of_lists):
    out = []
    for lst in iterable_of_lists:
        out.extend(lst)
    return out


def _is_regex_end(text, i):
    """text[i] 是否为正则句末标点（英文句点复用 sentence_segmenter 的保护判定）。"""
    ch = text[i]
    if ch in _REGEX_END_NOPROTECT:
        return True
    if ch == ".":
        return _is_english_period_end(text, i)
    return False
