"""app/pipeline/regex_segmenter.py 单元测试。

覆盖 --regex 后处理分句规则：标点正则切句（小数 3.14 / 连续点 ... / 英文句点保护）、
短句下限累积、长句上限强制切、VAD 上限按停顿切、VAD 下限合并、说话人不跨块、
[识别失败] / 无词级时间戳块原样透传、concat 不变量。
"""
from app.pipeline.regex_segmenter import regex_segment, _is_regex_end


def _w(text, s, e):
    return {"text": text, "start": s, "end": e}


def _seg(text, start, end, words, speaker=None):
    seg = {"text": text, "start": start, "end": end, "words": words}
    if speaker is not None:
        seg["speaker"] = speaker
    return seg


# 宽松默认：除被测维度外不触发其它切/并
_LOOSE = dict(long_sec=100.0, short_sec=0.3, vad_max_sec=100.0, vad_min_sec=0.0)


# ─── 标点正则切句 ─────────────────────────────────────────────────────

def test_sentence_punct_splits():
    seg = _seg("你好。再见。", 0.0, 4.0,
               [_w("你", 0, 1), _w("好", 1, 2), _w("再", 2, 3), _w("见", 3, 4)])
    out = regex_segment([seg], **_LOOSE)
    assert [s["text"] for s in out] == ["你好。", "再见。"]
    assert out[0]["start"] == 0.0 and out[0]["end"] == 2.0
    assert out[1]["start"] == 2.0 and out[1]["end"] == 4.0


def test_decimal_not_split():
    seg = _seg("值是3.14。", 0.0, 3.0,
               [_w("值", 0, 1), _w("是", 1, 1.5), _w("3.14", 1.5, 3)])
    out = regex_segment([seg], **_LOOSE)
    assert [s["text"] for s in out] == ["值是3.14。"]


def test_consecutive_dots_not_split():
    seg = _seg("等等...好的。", 0.0, 3.0,
               [_w("等", 0, 0.5), _w("等", 0.5, 1), _w("好", 2, 2.5), _w("的", 2.5, 3)])
    out = regex_segment([seg], **_LOOSE)
    assert [s["text"] for s in out] == ["等等...好的。"]


def test_english_period_splits_before_capital():
    seg = _seg("Google.You know.", 0.0, 3.0,
               [_w("Google", 0, 1), _w("You", 1.5, 2), _w("know", 2, 3)])
    out = regex_segment([seg], **_LOOSE)
    assert [s["text"] for s in out] == ["Google.", "You know."]


def test_is_regex_end_excludes_semicolon():
    # 用户指定标点集仅 .。?？!！，不含分号
    assert _is_regex_end("好；", 1) is False
    assert _is_regex_end("好。", 1) is True
    assert _is_regex_end("ok!", 2) is True


# ─── 短句下限：累积到 short 才切 ───────────────────────────────────────

def test_short_units_accumulate_until_short():
    seg = _seg("好。妙。", 0.0, 4.0,
               [_w("好", 0, 2), _w("妙", 2, 4)])
    out = regex_segment([seg], long_sec=100.0, short_sec=3.0,
                        vad_max_sec=100.0, vad_min_sec=0.0)
    # "好。" 仅 2s < 3s → 不切，并入下一单元
    assert [s["text"] for s in out] == ["好。妙。"]


# ─── 只在句末标点处切：无内部标点的长句保持完整 ───────────────────────

def test_long_sentence_without_punct_stays_whole():
    # 超长但句内无句末标点（且有大停顿）→ 绝不在停顿/非标点处切，保持完整
    seg = _seg("前半后半", 0.0, 5.0,
               [_w("前", 0, 1), _w("半", 1, 2), _w("后", 2.5, 3.5), _w("半", 3.5, 5)])
    out = regex_segment([seg], long_sec=3.0, short_sec=0.3,
                        vad_max_sec=2.0, vad_min_sec=0.0)
    assert [s["text"] for s in out] == ["前半后半"]


# ─── VAD 上限：仅在内部句末标点处再切（不在非标点处切）─────────────────

def test_vad_max_resplits_at_internal_punct():
    # 短句被并到一起形成 >vad_max 的合并句；vad_max 在内部句末标点处把它切回
    seg = _seg("好。妙。极佳的世界。", 0.0, 6.0,
               [_w("好", 0, 1), _w("妙", 1, 2),
                _w("极", 2, 3), _w("佳", 3, 4), _w("的", 4, 5),
                _w("世", 5, 5.5), _w("界", 5.5, 6)])
    # short=4 → "好。"(1s)、"妙。"(累计2s) 持续并入，整体并成 "好。妙。极佳的世界。"(6s)
    out = regex_segment([seg], long_sec=100.0, short_sec=4.0,
                        vad_max_sec=3.0, vad_min_sec=0.0)
    # vad_max=3：在内部句末标点处再切，每段 <=3s 且都断在 。 处
    assert [s["text"] for s in out] == ["好。妙。", "极佳的世界。"]
    assert all(s["text"].rstrip()[-1] in "。！？!?" for s in out)


def test_vad_max_single_sentence_no_internal_punct_stays_whole():
    # 单个完整长句（内部无句末标点）超过 vad_max → 无处可切，保持完整
    seg = _seg("这是一句没有内部句末标点的很长的话。", 0.0, 8.0,
               [_w(c, i * 0.4, i * 0.4 + 0.4) for i, c in
                enumerate("这是一句没有内部句末标点的很长的话")])
    out = regex_segment([seg], long_sec=100.0, short_sec=0.3,
                        vad_max_sec=3.0, vad_min_sec=0.0)
    assert len(out) == 1
    assert out[0]["text"] == seg["text"]


# ─── 成对收尾引号随句末归入前句（." / 。"）──────────────────────────────

def test_closing_quote_kept_with_sentence():
    seg = _seg("他说“你好。”然后呢？", 0.0, 5.0,
               [_w("他", 0, 0.5), _w("说", 0.5, 1), _w("你", 1.5, 2), _w("好", 2, 2.5),
                _w("然", 3, 3.5), _w("后", 3.5, 4), _w("呢", 4, 5)])
    out = regex_segment([seg], **_LOOSE)
    assert [s["text"] for s in out] == ["他说“你好。”", "然后呢？"]


def test_english_period_before_closing_quote_splits():
    seg = _seg('She said "ok." Then left.', 0.0, 4.0,
               [_w("She", 0, 0.5), _w("said", 0.5, 1), _w("ok", 1.2, 1.6),
                _w("Then", 2, 2.5), _w("left", 2.5, 4)])
    out = regex_segment([seg], **_LOOSE)
    assert out[0]["text"].rstrip() == 'She said "ok."'


# ─── VAD 下限：过短句合并 ─────────────────────────────────────────────

def test_vad_min_merges_short_into_prev():
    seg = _seg("你好世界。嗯。", 0.0, 3.5,
               [_w("你", 0, 0.7), _w("好", 0.7, 1.5), _w("世", 1.5, 2.2),
                _w("界", 2.2, 3.0), _w("嗯", 3.0, 3.5)])
    out = regex_segment([seg], long_sec=100.0, short_sec=0.3,
                        vad_max_sec=100.0, vad_min_sec=2.0)
    # "嗯。"(0.5s) < 2s → 并入前句
    assert [s["text"] for s in out] == ["你好世界。嗯。"]


def test_vad_min_merges_leading_short_into_next():
    seg = _seg("啊。好的世界。", 0.0, 4.0,
               [_w("啊", 0, 0.5), _w("好", 1.0, 1.8), _w("的", 1.8, 2.5),
                _w("世", 2.5, 3.2), _w("界", 3.2, 4.0)])
    out = regex_segment([seg], long_sec=100.0, short_sec=0.3,
                        vad_max_sec=100.0, vad_min_sec=2.0)
    assert [s["text"] for s in out] == ["啊。好的世界。"]
    assert out[0]["start"] == 0.0


# ─── 说话人 / 透传 ────────────────────────────────────────────────────

def test_no_merge_across_speaker_change():
    a = _seg("你好。", 0.0, 3.0,
             [_w("你", 0, 1.5), _w("好", 1.5, 3.0)], speaker="A")
    b = _seg("嗯。", 3.0, 3.5, [_w("嗯", 3.0, 3.5)], speaker="B")
    out = regex_segment([a, b], long_sec=100.0, short_sec=0.3,
                        vad_max_sec=100.0, vad_min_sec=2.0)
    assert [s["text"] for s in out] == ["你好。", "嗯。"]   # 不跨说话人合并
    assert out[0]["speaker"] == "A" and out[1]["speaker"] == "B"


def test_failure_marker_passthrough():
    a = _seg("你好。", 0.0, 2.0, [_w("你", 0, 1), _w("好", 1, 2)])
    fail = {"text": "[识别失败]", "start": 2.0, "end": 3.0, "words": None}
    b = _seg("再见。", 3.0, 4.0, [_w("再", 3, 3.5), _w("见", 3.5, 4)])
    out = regex_segment([a, fail, b], **_LOOSE)
    assert [s["text"] for s in out] == ["你好。", "[识别失败]", "再见。"]


def test_segments_without_words_passthrough_unchanged():
    segs = [{"text": "前半", "start": 0.0, "end": 2.0},
            {"text": "后半", "start": 2.0, "end": 4.0}]
    out = regex_segment([dict(s) for s in segs], **_LOOSE)
    assert out == segs


def test_concat_invariant():
    seg = _seg("你好世界。再见了。后会有期。", 0.0, 9.0,
               [_w("你", 0, 1), _w("好", 1, 2), _w("世", 2, 3), _w("界", 3, 4),
                _w("再", 4.5, 5), _w("见", 5, 5.5), _w("了", 5.5, 6),
                _w("后", 6.5, 7), _w("会", 7, 7.5), _w("有", 7.5, 8), _w("期", 8, 9)])
    out = regex_segment([seg], long_sec=4.0, short_sec=1.0,
                        vad_max_sec=3.0, vad_min_sec=1.0)
    assert "".join(s["text"] for s in out) == seg["text"]
