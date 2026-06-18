"""
profile_pipeline.py — 逐阶段性能剖析（Apple Silicon / CPU + OpenVINO 模式）

为 C++/MLX/Metal 迁移评估提供各组成部分的耗时基线。
通过 monkey-patch 在 ASRPipeline 各阶段插桩计时，跑完整离线流程（带说话人识别）。

用法（在装有 venv + models 的 asr-service 目录下运行）：
    venv/bin/python scripts/profile_pipeline.py /path/a.wav /path/b.wav
"""
import os
import sys
import time
import tempfile
import logging
from collections import defaultdict

logging.basicConfig(level=logging.WARNING)
for noisy in ("modelscope", "funasr", "torch", "root"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

import soundfile as sf

import app.config as cfg
from app.engines.vad_engine import VADEngine
from app.engines.openvino_asr_engine import OpenVINOASREngine
from app.engines.punc_engine import PuncEngine
from app.engines.speaker_embedding_engine import SpeakerEmbeddingEngine
from app.runtime.speaker_store import SpeakerStore
from app.runtime.speaker_service import SpeakerService
from app.pipeline.asr_pipeline import ASRPipeline
import app.pipeline.asr_pipeline as pipe_mod
import app.runtime.speaker_cluster as cluster_mod

# ─── 计时插桩 ───
TIMINGS = defaultdict(lambda: {"total": 0.0, "count": 0})


def timed(name, fn):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            TIMINGS[name]["total"] += dt
            TIMINGS[name]["count"] += 1
    return wrapper


def reset_timings():
    TIMINGS.clear()


def audio_duration(path):
    return sf.info(path).duration


def load_engines():
    print(">>> 加载模型 ...", flush=True)
    load_times = {}

    t = time.perf_counter()
    vad = VADEngine(); vad.load()
    load_times["VAD load"] = time.perf_counter() - t

    t = time.perf_counter()
    asr = OpenVINOASREngine(model_size="0.6b"); asr.load()
    load_times["ASR(OpenVINO) load"] = time.perf_counter() - t

    t = time.perf_counter()
    punc = PuncEngine(); punc.load()
    load_times["Punc load"] = time.perf_counter() - t

    t = time.perf_counter()
    spk = SpeakerEmbeddingEngine(); spk.load()
    load_times["Speaker load"] = time.perf_counter() - t

    db_path = os.path.join(tempfile.mkdtemp(), "speakers.db")
    store = SpeakerStore(db_path, model_tag=SpeakerEmbeddingEngine.MODEL_TAG)
    svc = SpeakerService(store, spk, vad)

    pipeline = ASRPipeline(asr, vad, punc, speaker_engine=spk, speaker_service=svc)
    return pipeline, asr, vad, punc, spk, svc, load_times


def instrument(pipeline, asr, vad, punc, spk, svc):
    # 0. ffmpeg 转换（pipeline 命名空间内引用）
    pipe_mod.convert_to_wav = timed("0. ffmpeg 转码", pipe_mod.convert_to_wav)
    # 1. VAD
    vad.detect = timed("1. VAD 切片", vad.detect)
    # 2. 切片落盘（含 soundfile 读写）
    pipeline._split_segments_to_chunks = timed(
        "2. 切片/重采样落盘", pipeline._split_segments_to_chunks)
    # 3. ASR（CPU OpenVINO 走逐条 transcribe）
    asr.transcribe = timed("3. ASR 识别", asr.transcribe)
    # 4. 标点
    punc.restore = timed("4. 标点恢复", punc.restore)
    # 4.5 说话人 embedding
    spk.embed_windows = timed("5. 说话人 embedding", spk.embed_windows)
    # 4.5 聚类（_run_diarization 内延迟 import，patch 模块属性）
    cluster_mod.cluster_offline = timed("6. 说话人聚类", cluster_mod.cluster_offline)
    # 4.6 声纹识别/登记
    svc.map_and_enroll_clusters = timed("7. 声纹识别/登记", svc.map_and_enroll_clusters)


def run_one(pipeline, audio_path, label):
    # pipeline 会删除输入文件，复制到临时目录保护原始样本
    import shutil
    tmp = os.path.join(tempfile.mkdtemp(), os.path.basename(audio_path))
    shutil.copy(audio_path, tmp)

    dur = audio_duration(audio_path)
    reset_timings()
    t0 = time.perf_counter()
    result = pipeline.run(
        tmp,
        task_id=f"prof_{label}",
        language=None,
        identify_speakers=True,
        options={"with_punc": True, "with_words": True, "diarize": True},
    )
    wall = time.perf_counter() - t0

    print(f"\n{'='*64}")
    print(f"文件: {audio_path}")
    print(f"音频时长: {dur:.1f}s | 总耗时(wall): {wall:.2f}s | RTF: {wall/dur:.3f}")
    n_seg = len(result.get("segments", []))
    spk_res = result.get("speakers", [])
    print(f"片段数: {n_seg} | 说话人数: {len(spk_res)} | 全文长度: {len(result.get('full_text',''))} 字")
    print(f"{'-'*64}")
    print(f"{'阶段':<24}{'耗时(s)':>10}{'占比':>8}{'调用次数':>8}")
    measured = sum(v["total"] for v in TIMINGS.values())
    for name in sorted(TIMINGS.keys()):
        v = TIMINGS[name]
        pct = 100 * v["total"] / wall
        print(f"{name:<24}{v['total']:>10.2f}{pct:>7.1f}%{v['count']:>8}")
    other = wall - measured
    print(f"{'(其他/调度开销)':<24}{other:>10.2f}{100*other/wall:>7.1f}%")
    print(f"{'='*64}")
    return {"file": audio_path, "duration": dur, "wall": wall,
            "rtf": wall/dur, "segments": n_seg, "speakers": len(spk_res),
            "stages": {k: dict(v) for k, v in TIMINGS.items()}}


def main():
    files = sys.argv[1:]
    if not files:
        print("用法: python scripts/profile_pipeline.py <wav> [wav ...]")
        sys.exit(1)

    pipeline, asr, vad, punc, spk, svc, load_times = load_engines()
    print("\n模型加载耗时:")
    for k, v in load_times.items():
        print(f"  {k:<24}{v:>8.2f}s")

    instrument(pipeline, asr, vad, punc, spk, svc)

    # 预热（用第一个文件跑一次，稳定 OpenVINO/JIT，丢弃计时）
    print(f"\n>>> 预热运行（丢弃计时）: {files[0]}", flush=True)
    run_one(pipeline, files[0], "warmup")

    print("\n>>> 正式测量", flush=True)
    results = []
    for i, f in enumerate(files):
        results.append(run_one(pipeline, f, f"file{i}"))

    import json
    out = os.path.join(os.path.dirname(__file__), "profile_result.json")
    with open(out, "w") as fp:
        json.dump({"load_times": load_times, "runs": results}, fp,
                  ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
