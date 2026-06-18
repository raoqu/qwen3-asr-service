"""
mlx_validate.py — 用 mlx-qwen3-asr 转写样本，测 RTF，并与 golden（OpenVINO）对比文本。

用法：venv-mlx/bin/python scripts/mlx_validate.py golden_baseline.json a.wav b.wav
"""
import os
import re
import sys
import json
import time

import mlx_qwen3_asr as M


def norm(s: str) -> str:
    """归一化：去标点/空白，便于纯 ASR 文本（不计标点差异）对比。"""
    return re.sub(r"[\s，。、？！,.!?；;：:\"'“”‘’（）()\[\]【】—\-…]", "", s or "")


def cer(ref: str, hyp: str) -> float:
    """字符级编辑距离 / 参考长度（归一化后）。"""
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # Levenshtein（滚动数组）
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def main():
    golden_path, *files = sys.argv[1:]
    golden = json.load(open(golden_path)) if os.path.exists(golden_path) else {}

    print(">>> 加载 MLX 模型 Qwen3-ASR-0.6B (fp16) ...", flush=True)
    import mlx.core as mx
    t0 = time.perf_counter()
    model, cfg = M.load_model("Qwen/Qwen3-ASR-0.6B", dtype=mx.float16)
    print(f"    模型加载耗时(含首次下载): {time.perf_counter()-t0:.1f}s", flush=True)

    import wave
    def wav_duration(path):
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())

    out = {}
    for f in files:
        name = os.path.basename(f)
        dur = wav_duration(f)
        # 仅验证 MLX ASR 核心（说话人分离我们仍用 CAM++，不走 mlx 的 pyannote 路径）
        # 预热同文件一次（稳定 Metal kernel），再正式计时
        print(f"\n>>> [{name}] 预热 ...", flush=True)
        _ = M.transcribe(f, model=model, diarize=False, language=None)
        t0 = time.perf_counter()
        res = M.transcribe(f, model=model, diarize=False, language=None,
                           return_timestamps=False)
        wall = time.perf_counter() - t0
        text = res.text or ""
        n_spk = 0
        g = golden.get(name, {})
        gtext = g.get("full_text", "")
        c = cer(gtext, text) if gtext else None
        g_rtf = g.get("rtf")
        speedup = (g_rtf / (wall / dur)) if g_rtf else None
        out[name] = {"duration": round(dur, 2), "wall": round(wall, 2),
                     "rtf": round(wall / dur, 4), "chars": len(text),
                     "n_speakers": n_spk, "cer_vs_golden": (round(c, 4) if c is not None else None),
                     "golden_rtf": g_rtf, "speedup_vs_openvino": (round(speedup, 2) if speedup else None),
                     "text_head": text[:200]}
        print(f">>> [{name}] MLX RTF={wall/dur:.4f} (OpenVINO {g_rtf}) "
              f"加速 {speedup:.2f}x | {len(text)}字 | 说话人{n_spk} | CER vs golden={c}")
        print(f"    MLX : {text[:120]}")
        print(f"    GOLD: {gtext[:120]}")

    dst = os.path.join(os.path.dirname(__file__), "mlx_validate_result.json")
    json.dump(out, open(dst, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved {dst}")


if __name__ == "__main__":
    main()
