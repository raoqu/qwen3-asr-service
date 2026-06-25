"""正则后处理分句（--regex）。

当 cfg.REGEX_SEGMENT 开启且对齐器启用（segment 带词级时间戳 words）时，对默认
segment_sentences 的输出再做一遍「标点正则 + 时长约束」重切，替换其结果；未开启时
管线行为与现状完全一致（本模块不被调用）。

核心约束——**所有句子边界只落在完整的句末标点处**：
切点只允许出现在句末标点 .。?？!！ 之后（含其后紧跟的成对收尾引号/括号，如 ." 。"），
绝不在停顿/时间等非标点位置切句。找不到句末标点的超长连续句保持完整、不强行切开。

四参数（默认值见 config.py / 暴露见 arg_schema），全部只在句末标点边界上起作用：

  - short_sec（短句下限）：在句末标点边界，左侧累积句长不足此值则不切、继续并入下一句
    （把过短碎句并进来），合并不得超过 long_sec。
  - long_sec（合并硬上限）：累积/合并句子的时长硬上限，绝不跨越此值合并。
  - vad_max_sec（软上限）：超过此值的句子，在其**内部句末标点**处再切到 <= 此值；
    句内无句末标点（单个完整长句）则保持完整不切。
  - vad_min_sec（VAD 下限）：仍短于此值的句子并入相邻句（合并后不超过 long_sec）。

不跨说话人：相邻 segment 说话人不同、或 [识别失败] 标记块，均作为硬边界分块处理。
说话人最终标签由管线 4.7 步按新句界重算。

concat 不变量：输出各句文本顺序拼接 == 输入文本顺序拼接（只重排边界，不增删字符）。
"""
from app.pipeline.sentence_segmenter import (
    _word_positions, _spans, _pieces, _CLAUSE_PUNCT, _is_cjk,
)

_FAIL_MARK = "[识别失败]"
# 句末标点正则集合（用户指定）：中文 。？！ + 英文 ? ！(全角) ! ；英文 . 走保护判定。
_REGEX_END_NOPROTECT = "。？！?！!"
# 句末标点后紧跟的成对收尾符号（引号/括号）随句末一并归入前句，使切点落在 ." / 。" 之后。
_CLOSERS = "\"'””’」』）)】》〉］]"


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
    """对单个「同说话人、连续、带 words」的块整体重切（切点只在句末标点处）。"""
    full_text = "".join(s["text"] for s in block)
    words = []
    for s in block:
        words.extend(s.get("words") or [])
    speaker = block[0].get("speaker")
    if not full_text.strip() or not words:
        return list(block)

    positions = _word_positions(full_text, words)
    n = len(full_text)
    cut_ends = [c for c in _punct_cut_ends(full_text) if 0 < c < n]
    spans = _spans(0, n, cut_ends)
    pieces = _pieces(full_text, words, positions,
                     float(block[0]["start"]), float(block[-1]["end"]), spans, speaker)

    sentences = _accumulate(pieces, short_sec, long_sec)            # 标点边界累积（short/long）
    sentences = _flatten(_split_overlong(s, vad_max_sec)           # 软上限：内部句末标点处再切
                         for s in sentences)
    sentences = _merge_short(sentences, vad_min_sec, long_sec)     # VAD 下限：过短并入相邻

    return [_finalize(s) for s in sentences if (s.get("text") or "").strip()]


# ─── 标点边界累积（short 下限 / long 硬上限）──────────────────────────

def _accumulate(pieces, short_sec, long_sec):
    """逐句末标点边界决定切/并：左侧累积句长 >= short_sec 即在此切；不足则并入下一句
    （合并后不得超过 long_sec）。无词的尾随标点片段并入前句。"""
    sentences = []
    cur = None
    for p in pieces:
        if cur is None:
            cur = _copy(p)
            continue
        combined = float(p["end"]) - float(cur["start"])
        if (not p.get("words") or _dur(cur) < short_sec) and combined <= long_sec:
            _merge_into(cur, p)
        else:
            sentences.append(cur)
            cur = _copy(p)
    if cur is not None:
        sentences.append(cur)
    return sentences


# ─── 软上限：仅在内部句末标点处再切 ───────────────────────────────────

def _split_overlong(s, max_sec):
    """时长 > max_sec 的句子，在其内部句末标点边界处切到 <= max_sec。

    若句内没有内部句末标点（单个完整长句），保持完整不切——绝不在非标点处下刀。
    """
    words = s.get("words")
    if _dur(s) <= max_sec or not words:
        return [s]
    text = s["text"]
    positions = _word_positions(text, words)
    n = len(text)
    ends = [c for c in _punct_cut_ends(text) if 0 < c < n]
    if not ends:
        return [s]                                   # 无内部句末标点 → 保持完整
    # 选「左侧时长 <= max_sec 的最大切点」；首个边界已超 max_sec 则取首个边界。
    chosen = ends[0]
    for c in ends:
        left_end = _left_end_time(words, positions, c)
        if left_end is None:
            continue
        if left_end - float(s["start"]) <= max_sec:
            chosen = c
        else:
            break
    left, right = _split_at_char(s, chosen)
    return [left] + _split_overlong(right, max_sec)


def _punct_cut_ends(text):
    """text 中所有句末标点的切点（标点之后、连同其后紧跟的成对收尾引号/括号）。"""
    ends = []
    n = len(text)
    i = 0
    while i < n:
        if _is_regex_end(text, i):
            j = i + 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            ends.append(j)
            i = j
        else:
            i += 1
    return ends


def _left_end_time(words, positions, cut):
    """字符切点 cut 左侧（position < cut）末词的 end 时间；左侧无词返回 None。"""
    left_end = None
    for w, pos in zip(words, positions):
        if pos < cut:
            left_end = w["end"]
    return left_end


def _split_at_char(s, cut):
    """在字符下标 cut 处把句子切成两句；时间取词级时间戳。"""
    text, words = s["text"], s["words"]
    positions = _word_positions(text, words)
    lw = [w for w, pos in zip(words, positions) if pos < cut]
    rw = [w for w, pos in zip(words, positions) if pos >= cut]
    left = {"text": text[:cut], "words": lw or None,
            "start": s["start"], "end": (lw[-1]["end"] if lw else s["start"]),
            "speaker": s.get("speaker")}
    right = {"text": text[cut:], "words": rw or None,
             "start": (rw[0]["start"] if rw else s["end"]), "end": s["end"],
             "speaker": s.get("speaker")}
    return left, right


# ─── VAD 下限：过短句合并 ─────────────────────────────────────────────

def _merge_short(sentences, vad_min_sec, long_sec):
    """短于 vad_min_sec 的句子并入相邻句：优先并入前句，合并后不得超过 long_sec。"""
    out = []
    for s in sentences:
        s = _copy(s)
        if out and _dur(s) < vad_min_sec and (float(s["end"]) - float(out[-1]["start"])) <= long_sec:
            _merge_into(out[-1], s)                   # 并入前句
            continue
        out.append(s)
    # 首句过短且无前句可并：尝试并入其后句（不超 long_sec）
    if len(out) >= 2 and _dur(out[0]) < vad_min_sec \
            and (float(out[1]["end"]) - float(out[0]["start"])) <= long_sec:
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
    """text[i] 是否为正则句末标点。

    英文句点保留小数 3.14 / .env / 单字母缩写 e.g. 的保护，并把句点后紧跟的成对收尾
    引号/括号视作透明（如 know." 仍判句末），与 _punct_cut_ends 的收尾符吞并一致。
    """
    ch = text[i]
    if ch in _REGEX_END_NOPROTECT:
        return True
    if ch != "." or i == 0:
        return False
    prev = text[i - 1]
    j = i + 1                                   # 跨过成对收尾引号/括号再看后随字符
    while j < len(text) and text[j] in _CLOSERS:
        j += 1
    nxt = text[j] if j < len(text) else ""
    if prev in _CLAUSE_PUNCT:                   # 标点簇 ",." 仍按句末（受后随字符约束）
        return nxt == "" or nxt.isspace() or nxt.isupper() or _is_cjk(nxt)
    if not prev.isalnum():
        return False                            # .env / 连续点 / 句点前是空白
    if prev.isdigit() and nxt.isdigit():
        return False                            # 小数 3.14
    k = i - 1                                    # 单字母缩写保护（e.g. / i.e.）
    while k >= 0 and text[k].isalnum():
        k -= 1
    if (i - 1 - k) < 2 and prev.isascii() and prev.isalpha():
        return False
    return nxt == "" or nxt.isspace() or nxt.isupper() or _is_cjk(nxt)
