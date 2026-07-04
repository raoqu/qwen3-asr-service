"""app/pipeline/vad_merge.py 单元测试。

覆盖 VAD 段 → ASR chunk 的合并规则，重点复现并防护「落进静音空档的幽灵词」根因：
合并绝不桥接长静音，否则对齐器把词时间戳散布进静音区，导致段级时间戳回退乱序。
"""
from app.pipeline.vad_merge import merge_vad_segments, vad_voiced_duration_sec


def test_merges_across_short_gap():
    # 短停顿（0.5s <= max_gap）被桥接：碎片话语合并为一个 chunk
    merged = merge_vad_segments([(0, 3000), (3500, 4000)], max_span_sec=5, max_gap_sec=2.0)
    assert merged == [(0, 4000)]


def test_does_not_bridge_long_silence():
    # 复现幽灵词根因：两段真实语音被 35s 静音隔开。即便跨度上限很大足以容纳，
    # 也绝不并入同一 chunk——否则 [235k,270k] 的静音区会被对齐器填入本属 305k+ 的词。
    merged = merge_vad_segments(
        [(265_000, 270_000), (305_000, 320_000)],
        max_span_sec=1000, max_gap_sec=2.0,      # 跨度上限刻意放大，仅靠 gap 约束拦截
    )
    assert merged == [(265_000, 270_000), (305_000, 320_000)]


def test_gap_boundary_exact():
    # 间隙恰等于 max_gap → 合并（<=）；恰超一毫秒 → 切开
    assert merge_vad_segments([(0, 1000), (3000, 4000)], 100, 2.0) == [(0, 4000)]
    assert merge_vad_segments([(0, 1000), (3001, 4000)], 100, 2.0) == [(0, 1000), (3001, 4000)]


def test_span_cap_still_applies():
    # gap 很小但总跨度超上限 → 仍按跨度切组（两约束是「与」关系）
    merged = merge_vad_segments([(0, 4000), (4200, 9000)], max_span_sec=5, max_gap_sec=2.0)
    assert merged == [(0, 4000), (4200, 9000)]


def test_empty_and_single():
    assert merge_vad_segments([], 5, 2.0) == []
    assert merge_vad_segments([(0, 3000)], 5, 2.0) == [(0, 3000)]


def test_chain_merge_then_break():
    # 连续短间隙链式合并，遇长静音断开，之后继续合并
    segs = [(0, 1000), (1200, 2000), (2200, 3000),      # 链式合并 → (0,3000)
            (30_000, 31_000), (31_100, 32_000)]         # 长静音后另起 → (30000,32000)
    merged = merge_vad_segments(segs, max_span_sec=1000, max_gap_sec=2.0)
    assert merged == [(0, 3000), (30_000, 32_000)]


# ─── vad_voiced_duration_sec：句级 VAD 语音总时长 ────────────────────────

def test_voiced_sums_overlapping_segments():
    # 句子 [0,10]s 内含两段语音（0-2s、3-5s）与一段静音间隙 → 总语音 4s
    vad = [(0, 2000), (3000, 5000), (12_000, 15_000)]  # 第三段在句外，不计
    assert vad_voiced_duration_sec(0.0, 10.0, vad) == 2.0 + 2.0


def test_voiced_clamps_to_sentence_bounds():
    # VAD 段跨越句子边界 → 只计入落在句内的部分（保证 <= 句子跨度）
    vad = [(500, 8000)]                                 # 0.5s~8s
    voiced = vad_voiced_duration_sec(1.0, 5.0, vad)     # 句子 [1,5]s，交集 [1,5]=4s
    assert voiced == 4.0
    assert voiced <= 5.0 - 1.0                          # 恒不超过句子跨度


def test_voiced_never_exceeds_span_when_fully_voiced():
    # 整句全程发声：语音总时长恰等于句子跨度，不会超出
    vad = [(0, 100_000)]
    span = 7.0 - 2.0
    assert vad_voiced_duration_sec(2.0, 7.0, vad) == span


def test_voiced_empty_or_degenerate():
    assert vad_voiced_duration_sec(3.0, 3.0, [(0, 100_000)]) == 0.0   # 零时长句
    assert vad_voiced_duration_sec(5.0, 2.0, [(0, 100_000)]) == 0.0   # start > end
    assert vad_voiced_duration_sec(0.0, 10.0, []) == 0.0              # 无 VAD 段


def test_voiced_segment_entirely_outside():
    # 句子落在两段语音之间的静音里 → 语音总时长为 0
    vad = [(0, 1000), (8000, 9000)]
    assert vad_voiced_duration_sec(3.0, 6.0, vad) == 0.0
