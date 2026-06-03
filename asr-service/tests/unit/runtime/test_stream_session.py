"""app/runtime/stream_session.py 测试（mock svad/asr/punc，真实 executor + Semaphore）。

异步用例依赖 pytest-asyncio（asyncio_mode=auto）。验证按句 final 产出、seg_id 递增、
时间戳偏移、对齐 words、标点、长句兜底、flush 末句，以及 backend 并发准入与 AudioBuffer。
"""
import asyncio
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.engines.vad_engine import VADEngine
from app.runtime.stream_session import StreamSession, VadOfflineBackend, AudioBuffer


class FakeSVAD:
    """按 feed 调用序号返回脚本化事件；is_final 调用返回 final_events。"""

    def __init__(self, events_by_call=None, final_events=None):
        self.events_by_call = events_by_call or {}
        self.final_events = final_events or []
        self.calls = 0

    def new_cache(self):
        return {}

    def process_chunk(self, arr, cache, is_final):
        if is_final:
            return list(self.final_events)
        ev = self.events_by_call.get(self.calls, [])
        self.calls += 1
        return list(ev)


def _pcm_ms(ms, sr=16000):
    """生成 ms 毫秒的非零 PCM16 字节（int16）。"""
    n = int(ms * sr / 1000)
    return (np.ones(n, dtype="<i2") * 1000).tobytes()


def _make_session(svad, *, enable_words=False, punc=None, max_segment_sec=30,
                  asr_result=None):
    asr = MagicMock()
    asr.transcribe_array.return_value = asr_result or [types.SimpleNamespace(text="hi")]
    executor = ThreadPoolExecutor(max_workers=2)
    sem = asyncio.Semaphore(1)
    s = StreamSession("sid", svad, asr, punc, executor, sem,
                      enable_words=enable_words, max_segment_sec=max_segment_sec)
    s.configure({"audio_fs": 16000})
    return s, asr, executor


async def _collect(agen):
    return [m async for m in agen]


# ─── StreamSession ───

async def test_complete_event_emits_final():
    svad = FakeSVAD(events_by_call={0: [{"type": "complete", "start": 0, "end": 1000}]})
    s, asr, ex = _make_session(svad)
    try:
        msgs = await _collect(s.feed_audio(_pcm_ms(1000)))
        assert len(msgs) == 1
        m = msgs[0]
        assert m["type"] == "final" and m["seg_id"] == 0
        assert m["text"] == "hi"
        assert m["start"] == 0 and m["end"] == 1000
        assert "words" not in m
    finally:
        ex.shutdown(wait=False)


async def test_start_then_end_emits_final():
    svad = FakeSVAD(events_by_call={
        0: [{"type": "start", "start": 0, "end": None}],
        1: [{"type": "end", "start": None, "end": 1000}],
    })
    s, asr, ex = _make_session(svad)
    try:
        assert await _collect(s.feed_audio(_pcm_ms(500))) == []     # 仅 start，无输出
        msgs = await _collect(s.feed_audio(_pcm_ms(500)))           # end → final
        assert len(msgs) == 1
        assert msgs[0]["start"] == 0 and msgs[0]["end"] == 1000
    finally:
        ex.shutdown(wait=False)


async def test_seg_id_increments_across_finals():
    svad = FakeSVAD(events_by_call={
        0: [{"type": "complete", "start": 0, "end": 500}],
        1: [{"type": "complete", "start": 500, "end": 1000}],
    })
    s, asr, ex = _make_session(svad)
    try:
        m0 = await _collect(s.feed_audio(_pcm_ms(500)))
        m1 = await _collect(s.feed_audio(_pcm_ms(500)))
        assert m0[0]["seg_id"] == 0
        assert m1[0]["seg_id"] == 1
    finally:
        ex.shutdown(wait=False)


async def test_flush_emits_pending_segment():
    # 收到 start 未收到 end，flush 时冲刷剩余缓冲
    svad = FakeSVAD(events_by_call={0: [{"type": "start", "start": 0, "end": None}]})
    s, asr, ex = _make_session(svad)
    try:
        await _collect(s.feed_audio(_pcm_ms(800)))
        flushed = await _collect(s.flush())
        assert len(flushed) == 1
        assert flushed[0]["type"] == "final"
        assert flushed[0]["start"] == 0
    finally:
        ex.shutdown(wait=False)


async def test_words_attached_when_enabled():
    word = types.SimpleNamespace(text="a", start_time=0.1, end_time=0.5)
    item = types.SimpleNamespace(text="a", time_stamps=types.SimpleNamespace(items=[word]))
    svad = FakeSVAD(events_by_call={0: [{"type": "complete", "start": 1000, "end": 2000}]})
    s, asr, ex = _make_session(svad, enable_words=True, asr_result=[item])
    try:
        msgs = await _collect(s.feed_audio(_pcm_ms(2000)))
        assert "words" in msgs[0]
        # 偏移叠加：start_ms=1000 -> +1.0s
        assert msgs[0]["words"][0] == {"text": "a", "start": 1.1, "end": 1.5}
    finally:
        ex.shutdown(wait=False)


async def test_punctuation_applied():
    punc = MagicMock()
    punc.restore.return_value = "hi。"
    svad = FakeSVAD(events_by_call={0: [{"type": "complete", "start": 0, "end": 1000}]})
    s, asr, ex = _make_session(svad, punc=punc)
    try:
        msgs = await _collect(s.feed_audio(_pcm_ms(1000)))
        assert msgs[0]["text"] == "hi。"
        punc.restore.assert_called_once()
    finally:
        ex.shutdown(wait=False)


async def test_long_segment_fallback_split():
    # start 后长时间无 end，超过 max_segment_sec 强制切分
    svad = FakeSVAD(events_by_call={0: [{"type": "start", "start": 0, "end": None}]})
    s, asr, ex = _make_session(svad, max_segment_sec=1)  # 1s 阈值
    try:
        msgs = await _collect(s.feed_audio(_pcm_ms(1500)))  # 1.5s > 1s
        assert len(msgs) == 1
        assert msgs[0]["type"] == "final"
        assert msgs[0]["start"] == 0
    finally:
        ex.shutdown(wait=False)


# ─── VadOfflineBackend ───

async def test_backend_acquire_limit_and_release():
    vad = VADEngine()
    vad._model = MagicMock()
    asr = MagicMock()
    asr.align_enabled = False
    backend = VadOfflineBackend(asr, vad, None, max_sessions=2, asr_concurrency=1)
    try:
        assert await backend.acquire() is True
        assert await backend.acquire() is True
        assert await backend.acquire() is False      # 超额
        backend.release(backend.create_session("x"))  # 释放一个
        assert await backend.acquire() is True
    finally:
        backend.shutdown()


def test_backend_capabilities_reflect_align():
    vad = VADEngine()
    vad._model = MagicMock()
    asr_aligned = MagicMock()
    asr_aligned.align_enabled = True
    b1 = VadOfflineBackend(asr_aligned, vad, None)
    assert b1.capabilities["word_timestamps"] is True
    b1.shutdown()

    asr_plain = MagicMock()
    asr_plain.align_enabled = False
    b2 = VadOfflineBackend(asr_plain, vad, None)
    assert b2.capabilities["word_timestamps"] is False
    b2.shutdown()


# ─── AudioBuffer ───

def test_audio_buffer_slice_and_drop():
    buf = AudioBuffer(16000)
    buf.append(np.arange(16000, dtype=np.float32))   # 0..1000ms
    buf.append(np.arange(16000, dtype=np.float32))   # 1000..2000ms
    assert buf.end_ms == 2000
    assert buf.slice_ms(0, 1000).shape[0] == 16000
    assert buf.slice_ms(1000, 2000).shape[0] == 16000

    buf.drop_until_ms(1000)
    assert buf.base_ms == 1000
    assert buf.slice_ms(1000, 2000).shape[0] == 16000
