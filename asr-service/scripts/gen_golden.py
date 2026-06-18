"""
gen_golden.py — 用当前（OpenVINO+CAM++）流水线产出 golden 输出，供 MLX 迁移精度回归。

保存每个样本的 full_text / segments(start,end,text,speaker) / speakers。
用法：cd asr-service && PYTHONPATH=. venv/bin/python scripts/gen_golden.py a.wav b.wav
"""
import os
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
from app.engines.openvino_asr_engine import OpenVINOASREngine
from app.engines.punc_engine import PuncEngine
from app.engines.speaker_embedding_engine import SpeakerEmbeddingEngine
from app.runtime.speaker_store import SpeakerStore
from app.runtime.speaker_service import SpeakerService
from app.pipeline.asr_pipeline import ASRPipeline


def build():
    vad = VADEngine(); vad.load()
    asr = OpenVINOASREngine(model_size="0.6b"); asr.load()
    punc = PuncEngine(); punc.load()
    spk = SpeakerEmbeddingEngine(); spk.load()
    db = os.path.join(tempfile.mkdtemp(), "speakers.db")
    svc = SpeakerService(SpeakerStore(db, model_tag=SpeakerEmbeddingEngine.MODEL_TAG), spk, vad)
    return ASRPipeline(asr, vad, punc, speaker_engine=spk, speaker_service=svc)


def main():
    files = sys.argv[1:]
    pipeline = build()
    out = {}
    for i, f in enumerate(files):
        tmp = os.path.join(tempfile.mkdtemp(), os.path.basename(f))
        shutil.copy(f, tmp)
        dur = sf.info(f).duration
        t0 = time.perf_counter()
        res = pipeline.run(tmp, task_id=f"golden_{i}", language=None,
                           identify_speakers=True,
                           options={"with_punc": True, "with_words": True, "diarize": True})
        wall = time.perf_counter() - t0
        segs = [{"start": round(s["start"], 3), "end": round(s["end"], 3),
                 "text": s["text"], "speaker": s.get("speaker")} for s in res["segments"]]
        out[os.path.basename(f)] = {
            "backend": "openvino_int8_0.6b + campplus",
            "duration": round(dur, 2), "wall": round(wall, 2), "rtf": round(wall / dur, 3),
            "full_text": res["full_text"],
            "n_segments": len(segs),
            "speakers": [s if isinstance(s, str) else s.get("label") for s in res.get("speakers", [])],
            "segments": segs,
        }
        print(f"[golden] {os.path.basename(f)}: {len(segs)} segs, RTF {wall/dur:.3f}, "
              f"{len(res['full_text'])} chars")
    dst = os.path.join(os.path.dirname(__file__), "golden_baseline.json")
    with open(dst, "w") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
