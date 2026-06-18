"""
test_mlx_pipeline.py — 集成测试：完整 ASRPipeline 用 MLXASREngine 作 ASR 后端，
CAM++ 说话人分离照常启用。跑样本，比对 golden 文本(CER)，测端到端 RTF。

用法：PYTHONPATH=<worktree>/asr-service venv/bin/python scripts/test_mlx_pipeline.py golden.json a.wav ...
"""
import os
import re
import sys
import json
import time
import shutil
import tempfile
import logging

logging.basicConfig(level=logging.WARNING)
for n in ("modelscope", "funasr", "torch", "root"):
    logging.getLogger(n).setLevel(logging.ERROR)

import soundfile as sf
from app.engines.vad_engine import VADEngine
from app.engines.mlx_asr_engine import MLXASREngine
from app.engines.punc_engine import PuncEngine
from app.engines.speaker_embedding_engine import SpeakerEmbeddingEngine
from app.runtime.speaker_store import SpeakerStore
from app.runtime.speaker_service import SpeakerService
from app.pipeline.asr_pipeline import ASRPipeline


def norm(s):
    return re.sub(r"[\s，。、？！,.!?；;：:\"'“”‘’（）()\[\]【】—\-…]", "", s or "")


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def build():
    vad = VADEngine(); vad.load()
    asr = MLXASREngine(model_size="0.6b"); asr.load()
    punc = PuncEngine(); punc.load()
    spk = SpeakerEmbeddingEngine(); spk.load()
    db = os.path.join(tempfile.mkdtemp(), "speakers.db")
    svc = SpeakerService(SpeakerStore(db, model_tag=SpeakerEmbeddingEngine.MODEL_TAG), spk, vad)
    print(f"[engine] ASR backend = {asr.BACKEND}")
    return ASRPipeline(asr, vad, punc, speaker_engine=spk, speaker_service=svc)


def main():
    golden_path, *files = sys.argv[1:]
    golden = json.load(open(golden_path)) if os.path.exists(golden_path) else {}
    pipeline = build()

    for i, f in enumerate(files):
        name = os.path.basename(f)
        tmp = os.path.join(tempfile.mkdtemp(), name)
        shutil.copy(f, tmp)
        dur = sf.info(f).duration
        # 预热（首个文件预热 ASR/Metal kernel），再正式计时
        if i == 0:
            pipeline.run(shutil.copy(f, os.path.join(tempfile.mkdtemp(), name)),
                         task_id="warm", identify_speakers=True,
                         options={"with_punc": True, "with_words": True, "diarize": True})
        t0 = time.perf_counter()
        res = pipeline.run(tmp, task_id=f"mlx_{i}", language=None, identify_speakers=True,
                           options={"with_punc": True, "with_words": True, "diarize": True})
        wall = time.perf_counter() - t0
        gtext = golden.get(name, {}).get("full_text", "")
        g_rtf = golden.get(name, {}).get("rtf")
        c = cer(gtext, res["full_text"]) if gtext else None
        spks = res.get("speakers", [])
        nspk = len(spks)
        print(f"\n{'='*60}\n[{name}] dur={dur:.1f}s")
        print(f"  端到端: wall={wall:.2f}s RTF={wall/dur:.4f} (golden OpenVINO RTF {g_rtf})")
        if g_rtf:
            print(f"  端到端加速: {g_rtf/(wall/dur):.2f}x")
        print(f"  segments={res['n_segments'] if 'n_segments' in res else len(res['segments'])} "
              f"speakers={nspk} chars={len(res['full_text'])} CER_vs_golden={c}")
        print(f"  MLX : {res['full_text'][:110]}")
        print(f"  GOLD: {gtext[:110]}")


if __name__ == "__main__":
    main()
