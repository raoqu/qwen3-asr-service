"""app/pipeline/vad_merge.py 单元测试。

覆盖 VAD 段 → ASR chunk 的合并规则，重点复现并防护「落进静音空档的幽灵词」根因：
合并绝不桥接长静音，否则对齐器把词时间戳散布进静音区，导致段级时间戳回退乱序。
"""
from app.pipeline.vad_merge import merge_vad_segments


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
