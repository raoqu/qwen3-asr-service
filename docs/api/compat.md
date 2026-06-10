# 兼容接口（OpenAI / 阿里云 DashScope）

[← API v2 概览](v2.md) ｜ **中文** | [English](compat_EN.md)

为已接入 **OpenAI** 或 **阿里云 DashScope（Paraformer）** 语音生态的客户端提供 drop-in 兼容垫片：改写 SDK 的 base url 指向本服务的 `/compat/...` 前缀即可对接，无需改动业务代码。兼容层是适配垫片，复用现有转写管线与任务队列，与原生 `/v1`、`/v2` 完全隔离。

> 兼容接口默认**关闭**，需启动时显式开启：
> ```bash
> # 离线 + 实时（实时需同时 --enable-stream）
> python -m app.main --enable-openai-api --enable-dashscope-api --enable-stream --api-key sk-xxx
> ```

**设计原则**：诚实降级——上游协议里本服务不具备的能力（翻译、温度、引导词、热词、逐字增量等）一律**显式忽略并告警或报错，绝不静默伪造**。

## 目录

- [1. 客户端如何指向本服务](#1-客户端如何指向本服务)
- [2. 认证](#2-认证)
- [3. OpenAI 兼容](#3-openai-兼容)
  - [3.1 转写 `POST /audio/transcriptions`](#31-转写-post-audiotranscriptions)
  - [3.2 流式转写 `stream=true`（SSE）](#32-流式转写-streamtruesse)
  - [3.3 翻译 `POST /audio/translations`](#33-翻译-post-audiotranslations)
  - [3.4 模型清单 `GET /models`](#34-模型清单-get-models)
  - [3.5 实时转写 `WS /realtime`](#35-实时转写-ws-realtime)
- [4. DashScope 兼容](#4-dashscope-兼容)
  - [4.1 提交（异步）](#41-提交异步)
  - [4.2 轮询](#42-轮询)
  - [4.3 二跳转写结果](#43-二跳转写结果)
  - [4.4 实时识别 `WS /inference`](#44-实时识别-ws-inference)
- [5. 能力与限制速查](#5-能力与限制速查)
- [6. 错误码](#6-错误码)
- [7. 与原生 v2 的取舍](#7-与原生-v2-的取舍)

---

## 1. 客户端如何指向本服务

两套上游 SDK 都支持改写 base url——指到本服务的 `/compat/...` 前缀即可：

| 上游 | 配置项 | 指向 |
|------|--------|------|
| OpenAI Python SDK | `OpenAI(base_url=...)` | `http://<host>:8765/compat/openai/v1` |
| OpenAI 实时 | ws base | `ws://<host>:8765/compat/openai/v1` |
| DashScope SDK | `dashscope.base_http_api_url` | `http://<host>:8765/compat/dashscope/api/v1` |
| DashScope 实时 | `dashscope.base_websocket_api_url` | `ws://<host>:8765/compat/dashscope/api-ws/v1` |

前缀之后的上游子路径逐段原样保留（SDK 硬编码），故兼容路径与 `/v1`、`/v2` 零碰撞。

## 2. 认证

服务配置 `--api-key` 时，所有兼容端点要求 `Authorization: Bearer <api-key>`（与两套 SDK 默认携带方式一致）。实时 WS 也接受 query 参数 `?token=<api-key>`。未配置 api-key 时放行（不建议生产环境）。

---

## 3. OpenAI 兼容

客户端 base_url = `http://<host>:8765/compat/openai/v1`。

### 3.1 转写 `POST /audio/transcriptions`

同步转写：上传音频，**响应即返回**转写结果（内部走队列等待，超时上限 `--openai-sync-timeout`，默认 300s）。

**请求**（`multipart/form-data`）：

| 字段 | 必填 | 支持 | 说明 |
|------|------|------|------|
| `file` | 是 | ✅ | 音频文件，扩展名白名单同 v2 |
| `model` | 是 | 宽容 | 任意值；以服务实际加载模型为准（见 `GET /models`）|
| `language` | 否 | ✅ | ISO-639-1，如 `zh`/`en` |
| `response_format` | 否 | ✅ | `json`(默认)/`text`/`srt`/`verbose_json`/`vtt` |
| `timestamp_granularities[]` | 否 | ✅ | `word`/`segment`（需 `verbose_json`）；含 `word` 才返回词级时间戳 |
| `stream` | 否 | ✅ | `true` 走 SSE（见 [3.2](#32-流式转写-streamtruesse)）|
| `prompt` | 否 | ❌ 忽略 | 本服务不支持引导文本 |
| `temperature` | 否 | ❌ 忽略 | 本服务不支持采样温度 |

**响应**：

- `json`：`{"text": "识别全文"}`
- `text`：纯文本（`text/plain`）
- `verbose_json`：含 `segments[]`（`start`/`end`/`text` 为真值）与可选 `words[]`
  > ⚠️ `tokens`/`avg_logprob`/`compression_ratio`/`no_speech_prob`/`seek` 为**占位值**（本服务无对应数据），勿当真实置信度使用。
- `srt`/`vtt`：标准字幕文本（`text/plain`）

**示例**：
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8765/compat/openai/v1", api_key="sk-xxx")
r = client.audio.transcriptions.create(
    model="whisper-1", file=open("a.wav", "rb"),
    response_format="verbose_json", timestamp_granularities=["word", "segment"])
print(r.text)
```
```bash
curl http://localhost:8765/compat/openai/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-xxx" \
  -F file=@a.wav -F model=whisper-1 -F response_format=srt
```

### 3.2 流式转写 `stream=true`（SSE）

`POST /audio/transcriptions` 带 `stream=true` 走 Server-Sent Events，推 `transcript.text.delta` / `transcript.text.done`：

```
data: {"type": "transcript.text.delta", "delta": "你好"}

data: {"type": "transcript.text.delta", "delta": "世界"}

data: {"type": "transcript.text.done", "text": "你好世界"}
```

> 说明：本服务为整段离线解码——**先完整转写再分句吐字**，故首个 delta 的延迟≈整段转写时长（非边解码边吐 token）。`delta` 文本是最终结果的分块（无时间戳），`stream=true` 时 `response_format` 不适用。

### 3.3 翻译 `POST /audio/translations`

**不支持**。本服务为纯语音识别，无翻译能力。固定返回 `501`：
```json
{"error":{"message":"This service performs speech recognition only; translation is not supported.","type":"invalid_request_error","param":null,"code":"unsupported"}}
```

### 3.4 模型清单 `GET /models`

```json
{"object":"list","data":[{"id":"qwen3-asr-0.6b","object":"model","created":0,"owned_by":"qwen3-asr"}]}
```
`id` 反映服务实际加载的模型大小；请求侧 `model` 宽容接受任意值。

### 3.5 实时转写 `WS /realtime`

OpenAI Realtime transcription 会话（需 `--enable-stream`）。ws base = `ws://<host>:8765/compat/openai/v1`。

1. 服务端 → `session.created`（连接即下发）
2. 客户端 → `session.update`：配置语言/采样率（兼容 GA `session.audio.input` 与 beta 字段路径）
3. 客户端 → `input_audio_buffer.append`：`{"type":"...append","audio":"<base64 PCM16>"}`（持续）
4. 客户端 → `input_audio_buffer.commit`（或关闭连接）触发末句冲刷
5. 服务端 → 每句 `conversation.item.input_audio_transcription.completed`：`{"item_id":"item_0","transcript":"识别整句"}`

> **能力与限制**：当前实时后端为 VAD-offline，**只产整句 `…completed`，不产逐字 `…delta`**（`partial_results=false`）。逐字增量需后续 vLLM 流式后端。

---

## 4. DashScope 兼容

DashScope Paraformer **录音文件识别**（异步）：提交 → 轮询 → 取二跳结果。客户端 `dashscope.base_http_api_url = "http://<host>:8765/compat/dashscope/api/v1"`。

> ⚠️ DashScope 端点只接受 `file_urls`（URL 列表，服务端下载）。如需**本地文件上传**，请改用 OpenAI 端点 [`/audio/transcriptions`](#31-转写-post-audiotranscriptions)（multipart 上传）或原生 [`POST /v2/asr`](v2/transcription.md#提交-asr-任务)。

### 4.1 提交（异步）

`POST /services/audio/asr/transcription`，头 `X-DashScope-Async: enable`（必填，缺失 → 400）。

```json
{ "model":"paraformer-v2",
  "input":{"file_urls":["https://example.com/a.wav"]},
  "parameters":{"language_hints":["zh"],"diarization_enabled":false} }
```

| 参数 | 支持 | 映射 |
|------|------|------|
| `input.file_urls[]` | ✅ | 服务端下载（SSRF 防护）；每 URL 一个子任务；单请求 ≤16 个 |
| `parameters.language_hints[0]` | ✅ | → 识别语言 |
| `parameters.diarization_enabled` | ✅ | → 说话人分离 |
| `parameters.speaker_count` | ❌ 忽略 | 说话人数上限为服务级配置（`--speaker-max`），不支持按请求覆盖 |
| `parameters.channel_id` | ❌ | 单声道，固定 0 |
| `disfluency_removal_enabled` / `special_word_filter` / `timestamp_alignment_enabled` | ❌ 忽略 | 无对应能力 |

> ⚠️ `file_urls` 必须本服务**可访问**；默认禁止私网/回环地址（SSRF 防护），可用 `--compat-fetch-allow-private` 放开（仅内网可信环境）。

响应：`{"output":{"task_status":"PENDING","task_id":"<id>"},"request_id":"<rid>"}`

### 4.2 轮询

`POST|GET /tasks/{task_id}`：

```json
{ "output":{"task_id":"<id>","task_status":"SUCCEEDED",
    "results":[{"file_url":"https://…/a.wav",
                "transcription_url":".../tasks/<id>/transcription/0",
                "subtask_status":"SUCCEEDED"}],
    "task_metrics":{"TOTAL":1,"SUCCEEDED":1,"FAILED":0}} }
```
`task_status`：`PENDING`/`RUNNING`/`SUCCEEDED`/`FAILED`（多 file_urls 时聚合）。结果在 `transcription_url`（二跳）。

> 反代/容器部署时用 `--compat-external-base-url` 指定 `transcription_url` 外部基址；未配置时按 `X-Forwarded-Proto/Host` 或请求地址推导。
> 任务注册表仅内存保存（带 TTL）：服务重启后未取结果的 task_id 将查询不到（404），需重新提交。

### 4.3 二跳转写结果

`GET /tasks/{task_id}/transcription/{idx}`，时间单位**毫秒**：

```json
{ "file_url":"https://…/a.wav",
  "transcripts":[{"channel_id":0,"content_duration_in_milliseconds":8470,"text":"识别全文",
    "sentences":[{"begin_time":0,"end_time":3200,"text":"…","sentence_id":1,"speaker_id":0,
                  "words":[{"begin_time":0,"end_time":200,"text":"你","punctuation":""}]}]}] }
```

**示例**：
```python
import dashscope
dashscope.base_http_api_url = "http://localhost:8765/compat/dashscope/api/v1"
dashscope.api_key = "sk-xxx"
from dashscope.audio.asr import Transcription
task = Transcription.async_call(model="paraformer-v2",
        file_urls=["https://example.com/a.wav"], language_hints=["zh"])
result = Transcription.wait(task=task.output.task_id)   # SDK 内部轮询 + 取 transcription_url
print(result.output)
```

### 4.4 实时识别 `WS /inference`

Paraformer realtime（需 `--enable-stream`）。`header/payload` 信封：

1. 客户端 → `run-task`：`{"header":{"action":"run-task","task_id":"<uuid>","streaming":"duplex"},"payload":{"parameters":{"format":"pcm","sample_rate":16000,"language_hints":["zh"]}}}`
2. 服务端 → `task-started`
3. 客户端 → 二进制 PCM 帧（~100ms/帧）
4. 服务端 → 每句 `result-generated`：`payload.output.sentence` 含 `begin_time`/`end_time`(ms)/`text`/`sentence_end:true`/`words[]`
5. 客户端 → `finish-task` → 服务端 → `task-finished`
6. 同一连接可再发 `run-task` 起新任务（连接复用）

> **能力与限制**：同 OpenAI 实时——VAD-offline 只产整句（`sentence_end:true`），**不产中间结果**（`sentence_end:false`）。完整中间结果需后续 vLLM 流式后端。

---

## 5. 能力与限制速查

| 能力 | OpenAI 兼容 | DashScope 兼容 |
|------|------------|---------------|
| 离线转写 | ✅ transcriptions | ✅ 录音文件识别 |
| 本地文件上传 | ✅ multipart | ❌ 仅 file_urls（URL） |
| 词级时间戳 | ✅ verbose_json + word | ✅ words[] |
| 说话人分离 | ➖（OpenAI 无对应字段）| ✅ diarization_enabled |
| 翻译 | ❌ 501 | ➖ 不涉及 |
| HTTP 流式 | ✅ stream=true（SSE） | ➖ 不涉及 |
| 实时整句 | ✅ completed（需 --enable-stream）| ✅ result-generated（需 --enable-stream）|
| 实时逐字增量 | ❌ 需 vLLM | ❌ 需 vLLM |
| 置信度/logprob | ➖ 占位 | ➖ 不提供 |
| 引导/温度/热词/顺滑 | ❌ 忽略 | ❌ 忽略 |

## 6. 错误码

**OpenAI 风格**（`{"error":{"message","type","param","code"}}`）：

| HTTP | code | 场景 |
|------|------|------|
| 400 | `invalid_value` | response_format 非法等 |
| 401 | `invalid_api_key` | 鉴权失败 |
| 501 | `unsupported` | translations |
| 503 | `overloaded` | 任务队列已满 |
| 504 | `timeout` | 同步等待超 `--openai-sync-timeout` |
| 500 | `internal_error` | 转写失败 |

**DashScope 风格**（`{"code","message","request_id"}`）：

| HTTP | code | 场景 |
|------|------|------|
| 400 | `InvalidParameter` | 缺 X-DashScope-Async、file_urls 为空/超 16 个 |
| 401 | `InvalidApiKey` | 鉴权失败 |
| 404 | `UNKNOWN_TASK` | task_id 不存在/已过期/服务重启后丢失 |
| 子任务 | `FAILED` + code | 下载失败/SSRF 拒绝/转写失败/队列繁忙（`Throttling`）（在 `results[].subtask_status`）|

## 7. 与原生 v2 的取舍

- 需要**对接已有 OpenAI/DashScope 生态**（SDK、现成客户端）→ 用兼容接口。
- 需要本服务**全部能力**（声纹库、任务列表/取消、统一实时信封、按请求覆盖参数）→ 用原生 [API v2](v2.md)。
- 超长音频：OpenAI 同步端点受 `--openai-sync-timeout` 限制，超长建议走 DashScope 异步兼容或原生 v2 异步。
