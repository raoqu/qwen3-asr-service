"""result_parser.sanitize_words 单元测试。

阈值以真实故障签名校准（L09 讲座音频，见 asr.log 17:23 任务）：
- 词 end 超出 chunk 边界 2s（相对时间 20.1s > 音频 18.1s）→ 越界拒收
- 28 词被塌缩进 3.26s（8.6 词/秒，仅覆盖 19s chunk 的 17%）→ 低覆盖塌缩拒收
- 54 词被塌缩进 0.32s（169 词/秒）→ 硬塌缩拒收
"""
from app.utils.result_parser import sanitize_words


def _w(text, s, e):
    return {"text": text, "start": s, "end": e}


def test_normal_words_pass_unchanged():
    ws = [_w("a", 10.0, 10.3), _w("b", 10.3, 10.8), _w("c", 11.0, 11.4)]
    out, reason = sanitize_words(ws, 10.0, 20.0)
    assert reason is None
    assert [(w["start"], w["end"]) for w in out] == [(10.0, 10.3), (10.3, 10.8), (11.0, 11.4)]


def test_out_of_bounds_rejected():
    # 真实签名：chunk 262.6~280.7（18.1s），词 end 282.7 超界 2s
    ws = [_w("a", 271.1, 271.5), _w("b", 280.0, 282.7)]
    out, reason = sanitize_words(ws, 262.6, 18.1)
    assert out is None and "越界" in reason


def test_slight_overshoot_clamped_not_rejected():
    # 越界 <= 0.5s 容差：钳回边界，不拒收
    ws = [_w("a", 10.0, 10.5), _w("b", 29.8, 30.3)]
    out, reason = sanitize_words(ws, 10.0, 20.0)
    assert reason is None
    assert out[1]["end"] == 30.0


def test_order_broken_rejected():
    ws = [_w("a", 15.0, 15.5), _w("b", 11.0, 11.3)]   # 回退 4s
    out, reason = sanitize_words(ws, 10.0, 20.0)
    assert out is None and "词序" in reason


def test_hard_collapse_rejected():
    # 真实签名：54 词压进 0.32s
    ws = [_w(f"w{i}", 271.15 + i * 0.005, 271.15 + i * 0.005 + 0.005) for i in range(54)]
    out, reason = sanitize_words(ws, 262.6, 18.1)
    assert out is None and "塌缩" in reason


def test_low_coverage_collapse_rejected():
    # 真实签名：28 词压进 3.26s，chunk 19s（覆盖 17%、8.6 词/秒）
    ws = [_w(f"w{i}", 305.76 + i * 0.116, 305.76 + (i + 1) * 0.116) for i in range(28)]
    out, reason = sanitize_words(ws, 290.6, 19.0)
    assert out is None and "塌缩" in reason


def test_short_sentence_with_long_silence_passes():
    # 正常场景：20s chunk 只有开头 5s 有一句话（16 词、3.2 词/秒）→ 覆盖低但语速正常，不拒收
    ws = [_w(f"w{i}", 10.0 + i * 0.31, 10.0 + (i + 1) * 0.31) for i in range(16)]
    out, reason = sanitize_words(ws, 10.0, 20.0)
    assert reason is None and len(out) == 16


def test_fast_cjk_speech_passes():
    # 快速中文：18 字在 2.5s（7.2 字/秒）、chunk 3s → 覆盖高，不拒收
    ws = [_w("字", 5.0 + i * 0.139, 5.0 + (i + 1) * 0.139) for i in range(18)]
    out, reason = sanitize_words(ws, 5.0, 3.0)
    assert reason is None


def test_empty_returns_none():
    assert sanitize_words([], 0.0, 10.0) == (None, None)
    assert sanitize_words(None, 0.0, 10.0) == (None, None)


# ─── 完整性校验：对齐词数远少于文本词数（真实签名：60 词文本仅对齐 15 词）──

def test_incomplete_alignment_rejected():
    # 真实签名（2min 切片 chunk 22.6~40.7s）：文本 ~60 词、对齐器只产出 15 词
    ws = [_w(f"w{i}", 25.5 + i * 0.35, 25.5 + (i + 1) * 0.35) for i in range(15)]
    out, reason = sanitize_words(ws, 22.6, 18.1, expected_words=60)
    assert out is None and "不完整" in reason


def test_complete_alignment_passes():
    ws = [_w(f"w{i}", 10.0 + i * 0.4, 10.0 + (i + 1) * 0.4) for i in range(20)]
    out, reason = sanitize_words(ws, 10.0, 10.0, expected_words=21)
    assert reason is None and len(out) == 20


def test_short_text_not_checked_for_completeness():
    # 文本词数 < 8：不做完整性判定（统计误差大）
    ws = [_w("a", 1.0, 1.3)]
    out, reason = sanitize_words(ws, 0.0, 5.0, expected_words=4)
    assert reason is None


def test_count_content_words_latin_and_cjk():
    from app.utils.result_parser import count_content_words
    assert count_content_words("Hello world, it's fine.") == 4
    assert count_content_words("你好世界。") == 4
    assert count_content_words("mix 中文 and English 词") == 6
    assert count_content_words("") == 0


# ─── 词时长分布：混合塌缩（部分正常+部分微词）────────────────────────────

def test_mixed_collapse_micro_share_rejected():
    # 真实签名：13 词正常铺开(~300ms/词) + 47 词挤成微词(~8ms)，总体速率/覆盖擦边通过
    normal = [_w(f"n{i}", 25.5 + i * 0.31, 25.5 + i * 0.31 + 0.30) for i in range(13)]
    squeezed = [_w(f"s{i}", 29.6 + i * 0.008, 29.6 + i * 0.008 + 0.007) for i in range(47)]
    out, reason = sanitize_words(normal + squeezed, 22.6, 18.1, expected_words=60)
    assert out is None and "塌缩" in reason


def test_normal_micro_share_passes():
    # 好区实测：~5% 零时长词（'I'/'to' 等）不应触发拒收
    ws = [_w(f"w{i}", 10 + i * 0.3, 10 + i * 0.3 + 0.25) for i in range(19)]
    ws.append(_w("I", 15.7, 15.7))                     # 1/20 = 5% 微词
    out, reason = sanitize_words(ws, 10.0, 20.0, expected_words=20)
    assert reason is None and len(out) == 20
