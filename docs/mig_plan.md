# Apple Silicon C++ / MLX 性能优化与部署精简迁移计划

> 目标：在**技术选型不变**（仍是 Qwen3-ASR + FSMN-VAD + CT-Transformer 标点 + CAM++ 声纹 + 谱聚类分离）的前提下，
> 1. **优化性能** —— 充分利用 Apple Silicon 的 GPU（Metal / MLX），把现在 CPU-only 的推理迁到 GPU；
> 2. **简化部署、降低对 Python 框架的依赖** —— 用 C++ / MLX 取代 PyTorch / FunASR / OpenVINO 这套重型 Python 栈。
>
> 本文档第一部分给出**实测性能基线**（各组成部分耗时拆解），第二部分给出**可行性与优先级分析**，第三部分给出**分阶段迁移计划**。

---

## 〇、测量方法与环境

- **硬件**：Apple M3 Ultra（28 核 = 20 性能核 + 8 能效核），256 GB 统一内存，macOS 26.5.1。
- **当前运行模式**：Apple Silicon 无 CUDA → 走 **CPU 模式**，ASR 使用 **OpenVINO INT8（0.6B）**，VAD / 标点 / 声纹走 **PyTorch（CPU）**。
- **被测代码**：`feature/accurate-sentence-segmentation` 分支（a3f5d67，含新的分句逻辑与 `MAX_ASR_CHUNK_DURATION=20` 切块策略）。
- **测量手段**：`asr-service/scripts/profile_pipeline.py` —— 对 `ASRPipeline.run()` 各阶段 monkey-patch 插桩计时，跑完整离线流程，**开启说话人识别（`identify_speakers=True`，含声纹库登记/比对）**。预热 1 次后正式测量。
- **样本**：
  - `yuanzhuo.wav` —— 108.5s，多人对话（识别出 4 个说话人）；
  - `RAG_08min.wav` —— 476.5s，单人长音频（识别出 1 个说话人）。

> RTF（Real-Time Factor）= 处理耗时 / 音频时长，越小越快。

### 实测结果（端到端，含说话人识别）

| 阶段 | 技术栈（当前） | yuanzhuo (108.5s) | RAG_08min (476.5s) | 平均占比 |
|---|---|---:|---:|---:|
| 0. ffmpeg 转码 16k 单声道 | ffmpeg 子进程 | 0.19s | 0.23s | 0.2% |
| 1. VAD 切片 | FSMN-VAD / PyTorch CPU | 0.20s | 0.84s | 0.6% |
| 2. 切片/重采样落盘 | soundfile + numpy | 0.01s | 0.06s | <0.1% |
| **3. ASR 识别** | **OpenVINO INT8 (0.6B) CPU** | **19.96s** | **86.57s** | **~57%** |
| 4. 标点恢复 | CT-Transformer / PyTorch CPU | 2.35s | 6.69s | ~5–7% |
| **5. 说话人 embedding** | **CAM++ / PyTorch CPU** | **11.83s** | **57.55s** | **~35%** |
| 6. 说话人聚类 | scipy / scikit-learn | 0.02s | 0.03s | <0.1% |
| 7. 声纹识别/登记 | numpy + sqlite | 0.00s | 0.00s | <0.1% |
| 8. 分句（feature 分支新增） | 纯 Python 文本处理 | <0.03s（落入"其他"桶） | <0.03s | <0.1% |
| **端到端** | | **34.58s（RTF 0.319）** | **152.00s（RTF 0.319）** | 100% |

> 模型加载（一次性）：OpenVINO ASR ≈ 5.1s，VAD/标点/声纹 < 0.2s（标点首次运行需联网下载 ~278MB 模型，约 16 分钟，仅一次）。

### 关键发现

1. **两个阶段吃掉 ~92% 的时间：ASR（~57%）+ 说话人 embedding（~35%）。** 其余全部阶段加起来 < 8%。
2. **ASR-only RTF ≈ 0.18**（19.96/108.5、86.57/476.5 高度一致），是第一优化目标。
3. **说话人 embedding 是被低估的大头（RTF ≈ 0.11–0.12）。** CAM++ 在 CPU 上用 PyTorch 对每个 1.5s 滑窗（步长 0.75s）逐窗提取 FBank+TDNN，**耗时随语音时长线性增长**，8 分钟音频就花了 57.5s。这是第二优化目标，且常被忽视。
4. **标点（~5%）、VAD / ffmpeg / 切片 / 聚类 / 分句（合计 < 1.5%）** 都不是性能瓶颈。它们的迁移价值**不在性能，而在"砍依赖、简化部署"**（见第二部分）。
5. **`feature/accurate-sentence-segmentation` 分支的分句逻辑是纯 Python 文本处理，实测 < 0.03s，零性能影响**；它的迁移只是逻辑移植，不是性能项。但它**改变了 ASR 的切块策略**（连续语音段切分阈值由 5s 提到 20s），从而影响 ASR 每次调用的输入长度——本测量已基于该分支，故数据真实反映迁移后的目标形态。
6. **GPU 完全闲置。** M3 Ultra 拥有强大的 GPU 与 256GB 统一内存，当前 CPU-only 路径完全没用上——这正是 MLX/Metal 的最大空间。

---

## 一、可行性分析（逐组件）

### 生态前提：MLX 已有成熟的 Qwen3-ASR 实现

调研确认 Apple Silicon 上已有**生产可用、纯 MLX、无 PyTorch 依赖**的 Qwen3-ASR 实现，这极大降低了 ASR 迁移风险：

- **`mlx-qwen3-asr`**（github.com/moona3k/mlx-qwen3-asr）：从零用 MLX 重写官方 PyTorch 模型，**同权重、对齐参考输出**，Metal GPU 加速；支持 0.6B / 1.7B、fp16 / 8-bit / 4-bit 量化、流式（KV-cache）、词级时间戳（自带 forced aligner）、30 语言 + 22 中文方言。官方实测 M4 Pro 0.6B fp16 **RTF ≈ 0.08**，4-bit 在短音频上再快 ~4.7×。
- **`mlx-audio`**（github.com/Blaizzy/mlx-audio）：STT/TTS 库，内置 `qwen3_asr` 模型（0.6B/1.7B，8-bit），提供 `generate()` / `stream_transcribe()` / `align()` 接口与 Qwen3-ForcedAligner。
- **MLX 提供 C++ API**：意味着最终可做成**纯 C++、无 Python 运行时**的自包含二进制。

> 结论：ASR 这个最大头有现成的高质量 MLX 参考实现可直接对接/借鉴，不必从零造轮子。

### 逐组件可行性 / 收益 / 成本矩阵

| 组件 | 当前栈 | 性能占比 | 迁移目标 | 性能收益 | 砍依赖收益 | 工作量 | 风险 | 优先级 |
|---|---|---:|---|---|---|---|---|:--:|
| **ASR** | OpenVINO INT8 (0.6B) | **~57%** | **MLX (Metal GPU)** | **极高**（RTF 0.18→~0.04，GPU+量化） | 去 `openvino` | 中（有现成实现） | 低-中（精度需对齐校验） | **P0** |
| **说话人 embedding** | CAM++ / PyTorch | **~35%** | **MLX / Metal**（TDNN 移植） | **高**（RTF 0.11→~0.02，且可批处理滑窗） | 去 `torch`/`torchaudio` | 中（模型小但需手写 FBank+TDNN+Kaldi 对齐） | 中（FBank/CMN 须与现实现 bit 对齐，否则声纹库模板失效） | **P0/P1** |
| 标点 | CT-Transformer / FunASR | ~5% | MLX 或 ONNX Runtime | 低（已够快） | **去 `funasr` 一大依赖** | 中 | 低 | **P2** |
| VAD | FSMN-VAD / FunASR | <1% | ONNX Runtime(CoreML EP) 或 MLX | 可忽略 | **去 `funasr`/`torch`** | 中 | 低 | **P2** |
| 谱聚类 | scipy + scikit-learn | <1% | 纯 numpy 或 C++(Eigen) | 可忽略 | 去 `scipy`/`scikit-learn` | 低 | 低 | **P2** |
| 分句（feature 分支） | 纯 Python | <0.1% | C++ 逻辑移植 | 无 | 无新增依赖 | 低 | 低（边界用例多，需测试覆盖） | **P3**（随 C++ 化） |
| 音频预处理/Mel | ffmpeg + numpy | <1% | C++ / Accelerate vDSP | 可忽略 | 去 Python numpy 预处理 | 低-中 | 低 | **P3**（随 C++ 化） |
| 重采样/切片 IO | soundfile | <1% | C++(libsndfile/dr_wav) | 可忽略 | 去 `soundfile` | 低 | 低 | **P3** |

### 当前 Python 依赖全景（待精简）

重型依赖：`torch`、`torchaudio`、`funasr`、`openvino`、`scipy`、`scikit-learn`、`modelscope`、`soundfile`、`numpy`。
- ASR→MLX：去 `openvino`。
- 声纹→MLX：去 `torch`/`torchaudio`。
- 标点 + VAD→MLX/ONNX：去 `funasr`（连带去 `torch` 的最后用途）。
- 聚类→numpy/C++：去 `scipy`/`scikit-learn`。
- **理想终态**：仅 `mlx`(+`mlx-audio`) + `numpy` + `soundfile`，或进一步做成**纯 C++ 二进制**（MLX C++ + dr_wav + 自写 Mel/聚类），Python 运行时可选。

---

## 二、优先级与策略

排序原则：**性能收益 × 依赖精简收益 ÷ 风险与成本**。

- **P0（先做，收益最大）**
  1. **ASR → MLX**：吃掉 57% 耗时，有现成实现，去掉 OpenVINO。
  2. **说话人 embedding → MLX/Metal**：吃掉 35% 耗时，且随音频时长线性增长，长音频场景收益尤其大。
- **P1**：声纹链路收尾（embedding 与现声纹库模板 bit 对齐校验；不对齐则升 `MODEL_TAG` 重建库）。
- **P2（砍依赖为主）**：标点、VAD、聚类迁移——目的是把 `funasr`/`torch`/`scipy`/`sklearn` 整套移除，部署体积与启动时间大幅下降。
- **P3（部署形态终局）**：C++ 化预处理/分句/IO，做成自包含二进制，彻底去 Python。

### 为什么"说话人 embedding"必须和 ASR 并列 P0

很多迁移只盯着 ASR，但本实测显示 CAM++ 占了 1/3 时间。两点原因：
1. 它在 **CPU + PyTorch** 上逐窗算 Kaldi FBank（`torchaudio.compliance.kaldi`）+ TDNN 前向，没用 GPU；
2. 滑窗密度高（每 0.75s 一窗），8 分钟音频上百窗，**FBank 计算本身**就很重。
迁到 MLX/Metal 后可**整批滑窗一次性上 GPU**，预计 RTF 0.11→~0.02（5× 以上）。

---

## 三、分阶段迁移计划

### 阶段 A：建立可对比的基线与验证框架（前置，1–2 天）

- [x] 编写逐阶段性能剖析脚本（`asr-service/scripts/profile_pipeline.py`，已完成，产出本文档基线）。
- [ ] 固化**精度回归基准**：用现有 OpenVINO/PyTorch 流水线对两个样本（及一组短样本）产出 `segments` / `full_text` / `speakers` 作为 golden，后续每次迁移用 CER/WER + 说话人标签一致性 + 声纹相似度比对。
- [ ] 引入 `Engine` 接口的"双后端可切换"开关（已有引擎模式，新增 `--asr-backend mlx` 等），保证迁移可灰度、可回退。

### 阶段 B：ASR → MLX（P0，核心收益，1–2 周）

- [ ] 评估两条路线：(a) 直接依赖 `mlx-audio` 的 `qwen3_asr`；(b) 集成 `mlx-qwen3-asr`；(c) 参照其实现自研 `MLXASREngine`。推荐先 (a)/(b) 快速验证收益，再决定是否自研以贴合现有 `transcribe()/batch_transcribe()/transcribe_array()` 接口与流式会话。
- [ ] 新增 `app/engines/mlx_asr_engine.py`，实现与 `OpenVINOASREngine` 同构接口（含 0.6B/1.7B、language 映射、word timestamps）。
- [ ] 量化档位实测：fp16 / 8-bit / 4-bit 的 RTF × CER 折中，选默认档（建议 8-bit，"近 fp16 质量"）。
- [ ] 精度回归：对齐 golden（CER 漂移阈值，如 < 相对 2%）。
- [ ] 设备路由：`device.py` 增加 Apple Silicon 分支（`auto` 在 mac+MLX 可用时选 `mlx`）。
- **验收**：ASR-only RTF 从 ~0.18 降到 ~0.04–0.08；去除 `openvino` 依赖（保留为非 mac 平台后端）。

### 阶段 C：说话人 embedding → MLX/Metal（P0/P1，1–2 周）

- [ ] 用 MLX 重写 CAM++（FCM + TDNN + StatsPool），权重从现有 `campplus_cn_common.bin` 转换（torch state_dict → MLX）。
- [ ] **FBank 对齐**：用 MLX/Accelerate(vDSP) 复刻 `torchaudio.compliance.kaldi.fbank`（80 mel、25ms/10ms、dither=0）+ CMN，**与现实现数值对齐**（否则声纹库模板失效）。
- [ ] 滑窗批处理：整段所有窗一次性堆叠上 GPU。
- [ ] 声纹库兼容：若 embedding 与旧模板不能 bit 对齐 → 升 `SpeakerEmbeddingEngine.MODEL_TAG`，触发库重建（已有机制）。
- **验收**：说话人 embedding RTF 从 ~0.11 降到 ~0.02；去除 `torch`/`torchaudio`。

### 阶段 D：砍依赖（P2，标点 / VAD / 聚类，1–2 周）

- [ ] 标点 CT-Transformer：转 MLX 或 ONNX Runtime（CoreML EP），去 `funasr`。
- [ ] VAD FSMN：转 ONNX Runtime 或 MLX，去 `funasr` 最后用途 → **`funasr`/`torch` 整套移除**。
- [ ] 谱聚类 + AHC：scipy.linkage/eigsh + sklearn.KMeans → 纯 numpy 实现（linkage 用并查集、eigsh 用 numpy.linalg.eigh、KMeans 自写），去 `scipy`/`scikit-learn`。
- **验收**：requirements 仅剩 `mlx`(+`mlx-audio`)、`numpy`、`soundfile`；镜像/虚拟环境体积与启动时间显著下降。

### 阶段 E：C++ 终态（P3，可选，按需）

- [ ] 用 **MLX C++ API** 把 ASR + 声纹推理封进 C++ 核；预处理（Mel/FBank 用 Accelerate vDSP）、分句逻辑、聚类、WAV IO（dr_wav）全部 C++ 化。
- [ ] 对外保持现有 HTTP/WS API（C++ 服务，或 C++ 核 + 极薄 Python/FastAPI 壳）。
- **验收**：单一自包含二进制即可部署，无需 Python/conda 环境；冷启动毫秒级。

---

## 四、风险与注意事项

1. **精度对齐**：MLX 重实现必须对齐参考输出（ASR 的 MRoPE/窗口注意力细节、CAM++ 的 FBank/CMN）。务必先建 golden 回归再迁移。
2. **声纹库模板兼容**：embedding 任何数值变化都会让旧声纹库失配。策略：能 bit 对齐则平滑迁移；不能则升 `MODEL_TAG` 重建库（机制已存在）。
3. **平台分叉**：MLX 仅 Apple Silicon。需保留 OpenVINO/CUDA 后端给 Linux/Windows/服务器 GPU。迁移应是"新增 mac 高性能后端"，非替换。
4. **流式路径**：实时 WS 转写与离线共用 ASR 引擎，MLX 引擎需同时满足 `transcribe_array()` 与流式 KV-cache 语义。
5. **`feature/accurate-sentence-segmentation` 已并入主干**（见下），其切块策略（`MAX_ASR_CHUNK_DURATION=20`）应在 MLX ASR 引擎下复测——长块对 GPU 更友好，可能进一步拉低 RTF。
6. **量化质量**：4-bit 速度诱人但需逐语言核 CER；建议默认 8-bit，4-bit 作为可选档。

---

## 五、关于 `feature/accurate-sentence-segmentation` 分支

该分支新增/修改：
- `app/pipeline/sentence_segmenter.py`（新增，361 行，纯 Python 文本处理：按标点/停顿/说话人切换重组句子）；
- `app/pipeline/asr_pipeline.py`（+111 行，接入分句、调整切块与说话人标签精修顺序）；
- `app/config.py`（新增 `MAX_ASR_CHUNK_DURATION=20`、`SENTENCE_LONG_PAUSE_MS`、`SENTENCE_SHORT_PAUSE_MS`，解耦"处理切块"与"最终句子边界"）；
- `app/runtime/vllm_offline.py`（适配）；相关单测；`evolution.md`（设计说明 752 行）。

**对迁移计划的影响**：分句为纯 CPU 文本逻辑，零性能成本，C++ 化时仅作逻辑移植（阶段 E，需补足边界用例测试）。其切块策略改变了 ASR 输入长度，本基线已基于该分支测得，故迁移目标形态一致。

**合并状态**：本计划制定时已将 `feature/accurate-sentence-segmentation` 合并回 `main` 主干（无冲突，README 的中文简介保留 main 侧版本）。后续 MLX 迁移以合并后的 `main` 为起点。

---

## 附：一句话结论

> 把 **ASR（57%）和说话人 embedding（35%）这两块迁到 MLX/Metal**，预计端到端 RTF 从 **0.319 降到 ~0.07–0.10（约 3–5×）**；再把标点/VAD/聚类去 FunASR/Torch/scipy，最终可收敛到 **`mlx + numpy + soundfile`** 乃至**纯 C++ 自包含二进制**，同时实现"更快"与"更易部署"两个目标。ASR 已有成熟 MLX 参考实现，风险可控；最大工程难点是**声纹 FBank/embedding 的数值对齐**。
