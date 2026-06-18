"""
probe_speaker_backend.py — Stage C-1：CAM++ embedding 加速可行性探针。

目的：
  1. 用当前 PyTorch/CPU CAM++ 在真实滑窗上产出 reference embedding（后续 MLX 对齐基准）。
  2. 低风险方案探测：把 TDNN 前向搬到 Apple Metal（PyTorch MPS），
     测加速比，并逐元素比对 embedding 与 CPU 是否数值一致（决定是否真需 MLX 重写）。

用法：PYTHONPATH=<wt>/asr-service venv/bin/python scripts/probe_speaker_backend.py a.wav
"""
import os
import sys
import time
import logging

logging.basicConfig(level=logging.WARNING)
for n in ("modelscope", "funasr", "torch", "root"):
    logging.getLogger(n).setLevel(logging.ERROR)

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

from app.engines.speaker_embedding_engine import SpeakerEmbeddingEngine, make_windows
from app.engines.vad_engine import VADEngine


def build_feats(engine, wav, windows):
    """复刻 embed_windows 的特征准备：fbank + circle_pad + stack → [N, T, 80]。"""
    sr = engine.SAMPLE_RATE
    clips = []
    for st, ed in windows:
        clip = wav[max(int(st * sr), 0):int(ed * sr)]
        if len(clip) == 0:
            clip = np.zeros(1, dtype=np.float32)
        clips.append(torch.from_numpy(np.ascontiguousarray(clip)).float())
    max_len = max(max(c.shape[0] for c in clips), 400)
    feats = torch.stack([engine._fbank(engine._circle_pad(c, max_len)) for c in clips])
    return feats


def run_forward(model, feats, device, batch=64):
    model = model.to(device)
    feats_d = feats.to(device)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    outs = []
    with torch.no_grad():
        for i in range(0, len(feats_d), batch):
            outs.append(model(feats_d[i:i + batch]))
    emb = F.normalize(torch.cat(outs), dim=1)
    if device == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    return emb.cpu().numpy(), dt


def main():
    wav_path = sys.argv[1]
    print(f"torch {torch.__version__} | MPS available: {torch.backends.mps.is_available()}")

    eng = SpeakerEmbeddingEngine(); eng.load()
    vad = VADEngine(); vad.load()

    # 真实滑窗（VAD 段 → make_windows），贴合生产分离路径
    segs = vad.detect(wav_path)
    wav, _sr = sf.read(wav_path, dtype="float32")
    windows = [w for s, e in segs for w in make_windows(s / 1000.0, e / 1000.0)]
    print(f"音频 {sf.info(wav_path).duration:.1f}s → {len(segs)} VAD 段 → {len(windows)} 窗")

    feats = build_feats(eng, wav, windows)
    print(f"特征张量: {tuple(feats.shape)} (N, frames, mel)")

    model = eng._model

    # 预热 + 计时：CPU
    _ = run_forward(model, feats[:8], "cpu")
    emb_cpu, t_cpu = run_forward(model, feats, "cpu")
    print(f"\n[CPU ] {len(windows)} 窗前向: {t_cpu*1000:.0f} ms")

    if torch.backends.mps.is_available():
        try:
            _ = run_forward(model, feats[:8], "mps")  # 预热（含 kernel 编译）
            emb_mps, t_mps = run_forward(model, feats, "mps")
            # 数值比对
            cos = np.sum(emb_cpu * emb_mps, axis=1)  # 均已 L2 归一
            max_abs = float(np.max(np.abs(emb_cpu - emb_mps)))
            print(f"[MPS ] {len(windows)} 窗前向: {t_mps*1000:.0f} ms  → 加速 {t_cpu/t_mps:.2f}x")
            print(f"[对齐] cosine(CPU,MPS): min={cos.min():.6f} mean={cos.mean():.6f} | "
                  f"max|Δ|={max_abs:.2e}")
            print(f"[结论] {'embedding 数值一致（可直接用 MPS 加速，零声纹库风险）' if cos.min() > 0.9999 else 'embedding 有偏差，需谨慎（升 MODEL_TAG 或改用 MLX 并校准）'}")
        except Exception as e:
            print(f"[MPS ] 失败: {type(e).__name__}: {e}")
    else:
        print("[MPS ] 不可用")

    # 保存 CPU reference 供 MLX 对齐
    np.save(os.path.join(os.path.dirname(__file__), "speaker_ref_emb.npy"), emb_cpu)
    print(f"\nreference embedding 已保存 (shape {emb_cpu.shape})")


if __name__ == "__main__":
    main()
