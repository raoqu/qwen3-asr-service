"""app/engines/speaker_embedding_engine.py 单元测试（mock 模型 forward，不加载真权重）。

覆盖：make_windows 边界、_fbank 形状与 CMN、embed_windows 批切分与 L2 归一、
embed_segment 均值重归一化。精度类断言不在此（遵循实施方案 §4 铁律）。
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from app.engines.speaker_embedding_engine import (
    SpeakerEmbeddingEngine,
    make_windows,
)


# ─── make_windows ───

def test_make_windows_regular_sliding():
    assert make_windows(0.0, 3.0) == [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0)]


def test_make_windows_exact_one_window():
    assert make_windows(0.0, 1.5) == [(0.0, 1.5)]


def test_make_windows_tail_window_length_in_bounds():
    ws = make_windows(0.0, 1.6)
    assert ws == [(0.0, 1.5), (0.75, 1.6)]
    for st, ed in ws:
        assert 0.75 < ed - st <= 1.5


def test_make_windows_short_segment_patch():
    # 上游 chunk() 对 ≤0.75s 段产生 0 窗；本实现补丁为整段 1 窗
    assert make_windows(0.0, 0.5) == [(0.0, 0.5)]


def test_make_windows_with_offset():
    assert make_windows(10.0, 13.0) == [(10.0, 11.5), (10.75, 12.25), (11.5, 13.0)]


def test_make_windows_zero_length_segment():
    assert make_windows(2.0, 2.0) == []


# ─── _fbank ───

def test_fbank_shape_and_cmn():
    wav = torch.randn(16000)  # 1s
    feat = SpeakerEmbeddingEngine._fbank(wav)
    assert feat.shape[1] == 80
    assert feat.shape[0] > 90  # 25ms 帧长 / 10ms 步长 → 约 98 帧
    # CMN：每维均值为 0
    assert torch.allclose(feat.mean(0), torch.zeros(80), atol=1e-4)


# ─── embed_windows ───

def _engine_with_mock(dim=192):
    eng = SpeakerEmbeddingEngine()
    model = MagicMock(side_effect=lambda feats: torch.ones(feats.shape[0], dim) * 3.0)
    eng._model = model
    return eng, model


def test_embed_windows_requires_loaded():
    eng = SpeakerEmbeddingEngine()
    with pytest.raises(RuntimeError):
        eng.embed_windows(np.zeros(16000, dtype=np.float32), [(0.0, 1.0)])


def test_embed_windows_empty_windows():
    eng, _ = _engine_with_mock()
    out = eng.embed_windows(np.zeros(16000, dtype=np.float32), [])
    assert out.shape == (0, 192)


def test_embed_windows_batching_and_l2_norm():
    eng, model = _engine_with_mock()
    wav = np.random.default_rng(0).normal(size=16000 * 2).astype(np.float32)
    windows = [(0.0, 1.5)] * 130  # 130 窗 → 64+64+2 三批
    out = eng.embed_windows(wav, windows)
    assert out.shape == (130, 192)
    assert model.call_count == 3
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embed_windows_short_clip_circle_padded():
    eng, model = _engine_with_mock()
    wav = np.random.default_rng(1).normal(size=16000).astype(np.float32)
    # 混合长短窗：补齐后批内帧数一致才能 stack 成功
    out = eng.embed_windows(wav, [(0.0, 1.0), (0.0, 0.3)])
    assert out.shape == (2, 192)
    feats = model.call_args[0][0]
    assert feats.shape[0] == 2


def test_embed_segment_mean_renormalized():
    eng, _ = _engine_with_mock()
    wav = np.random.default_rng(2).normal(size=16000 * 3).astype(np.float32)  # 3s → 3 窗
    emb = eng.embed_segment(wav)
    assert emb.shape == (192,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_model_tag_constant():
    # V 系列衔接面：模板兼容性标识必须存在且非空
    assert SpeakerEmbeddingEngine.MODEL_TAG == "campplus_cn_common@v1"
