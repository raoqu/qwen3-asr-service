# Compatibility APIs (OpenAI / Alibaba Cloud DashScope)

[← API v2 Overview](v2_EN.md) ｜ [中文](compat.md) | **English**

Drop-in compatibility shims for clients already built against the **OpenAI** or **Alibaba Cloud DashScope (Paraformer)** speech ecosystems: just point the SDK's base url at this service's `/compat/...` prefix — no business-code changes. The compat layer is an adapter shim that reuses the existing transcription pipeline and task queue, fully isolated from the native `/v1` and `/v2`.

> Compat APIs are **off by default**; enable them explicitly at startup:
> ```bash
> # offline + realtime (realtime also needs --enable-stream)
> python -m app.main --enable-openai-api --enable-dashscope-api --enable-stream --api-key sk-xxx
> ```

**Design principle**: honest degradation — capabilities this service lacks (translation, temperature, prompt, hotwords, per-token deltas, etc.) are **explicitly ignored/warned or rejected, never silently faked**.

## Table of Contents

- [1. Pointing clients at this service](#1-pointing-clients-at-this-service)
- [2. Authentication](#2-authentication)
- [3. OpenAI compatibility](#3-openai-compatibility)
  - [3.1 Transcription `POST /audio/transcriptions`](#31-transcription-post-audiotranscriptions)
  - [3.2 Streaming `stream=true` (SSE)](#32-streaming-streamtrue-sse)
  - [3.3 Translation `POST /audio/translations`](#33-translation-post-audiotranslations)
  - [3.4 Models `GET /models`](#34-models-get-models)
  - [3.5 Realtime `WS /realtime`](#35-realtime-ws-realtime)
- [4. DashScope compatibility](#4-dashscope-compatibility)
  - [4.1 Submit (async)](#41-submit-async)
  - [4.2 Poll](#42-poll)
  - [4.3 Second-hop transcript](#43-second-hop-transcript)
  - [4.4 Realtime `WS /inference`](#44-realtime-ws-inference)
- [5. Capabilities & limits](#5-capabilities--limits)
- [6. Error codes](#6-error-codes)
- [7. Compat vs native v2](#7-compat-vs-native-v2)

---

## 1. Pointing clients at this service

Both upstream SDKs support overriding the base url — point it at this service's `/compat/...` prefix:

| Upstream | Setting | Target |
|------|--------|------|
| OpenAI Python SDK | `OpenAI(base_url=...)` | `http://<host>:8765/compat/openai/v1` |
| OpenAI realtime | ws base | `ws://<host>:8765/compat/openai/v1` |
| DashScope SDK | `dashscope.base_http_api_url` | `http://<host>:8765/compat/dashscope/api/v1` |
| DashScope realtime | `dashscope.base_websocket_api_url` | `ws://<host>:8765/compat/dashscope/api-ws/v1` |

Sub-paths after the prefix are preserved verbatim (hardcoded by the SDKs), so compat paths never collide with `/v1` or `/v2`.

## 2. Authentication

When the service is started with `--api-key`, all compat endpoints require `Authorization: Bearer <api-key>` (matching both SDKs' default behaviour). Realtime WS also accepts the query param `?token=<api-key>`. Without an api-key, requests pass through (not recommended in production).

---

## 3. OpenAI compatibility

Client base_url = `http://<host>:8765/compat/openai/v1`.

### 3.1 Transcription `POST /audio/transcriptions`

Synchronous transcription: upload audio, **the response returns the result** (internally waits on the queue, capped by `--openai-sync-timeout`, default 300s).

**Request** (`multipart/form-data`):

| Field | Required | Supported | Notes |
|------|------|------|------|
| `file` | yes | ✅ | Audio file; same extension allowlist as v2 |
| `model` | yes | lenient | Any value; the actually-loaded model is used (see `GET /models`) |
| `language` | no | ✅ | ISO-639-1, e.g. `zh`/`en` |
| `response_format` | no | ✅ | `json` (default)/`text`/`srt`/`verbose_json`/`vtt` |
| `timestamp_granularities[]` | no | ✅ | `word`/`segment` (needs `verbose_json`); include `word` for word-level timestamps |
| `stream` | no | ✅ | `true` switches to SSE (see [3.2](#32-streaming-streamtrue-sse)) |
| `prompt` | no | ❌ ignored | Guiding text not supported |
| `temperature` | no | ❌ ignored | Sampling temperature not supported |

**Response**:

- `json`: `{"text": "full transcript"}`
- `text`: plain text (`text/plain`)
- `verbose_json`: includes `segments[]` (`start`/`end`/`text` are real) and optional `words[]`
  > ⚠️ `tokens`/`avg_logprob`/`compression_ratio`/`no_speech_prob`/`seek` are **placeholders** (no corresponding data); do not treat them as real confidence.
- `srt`/`vtt`: standard subtitle text (`text/plain`)

**Example**:
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

### 3.2 Streaming `stream=true` (SSE)

`POST /audio/transcriptions` with `stream=true` uses Server-Sent Events, emitting `transcript.text.delta` / `transcript.text.done`:

```
data: {"type": "transcript.text.delta", "delta": "Hello"}

data: {"type": "transcript.text.delta", "delta": " world"}

data: {"type": "transcript.text.done", "text": "Hello world"}
```

> Note: this service decodes the whole clip offline — it **transcribes fully first, then emits sentence by sentence**, so the first delta's latency ≈ full transcription time (not token-by-token while decoding). `delta` text is a chunk of the final result (no timestamps); `response_format` does not apply when `stream=true`.

### 3.3 Translation `POST /audio/translations`

**Not supported**. This service is speech recognition only. Always returns `501`:
```json
{"error":{"message":"This service performs speech recognition only; translation is not supported.","type":"invalid_request_error","param":null,"code":"unsupported"}}
```

### 3.4 Models `GET /models`

```json
{"object":"list","data":[{"id":"qwen3-asr-0.6b","object":"model","created":0,"owned_by":"qwen3-asr"}]}
```
`id` reflects the actually-loaded model size; the request-side `model` accepts any value leniently.

### 3.5 Realtime `WS /realtime`

OpenAI Realtime transcription session (needs `--enable-stream`). ws base = `ws://<host>:8765/compat/openai/v1`.

1. Server → `session.created` (sent on connect)
2. Client → `session.update`: configure language/sample rate (accepts GA `session.audio.input` and beta field paths)
3. Client → `input_audio_buffer.append`: `{"type":"...append","audio":"<base64 PCM16>"}` (continuous)
4. Client → `input_audio_buffer.commit` (or close) flushes the trailing sentence
5. Server → per sentence `conversation.item.input_audio_transcription.completed`: `{"item_id":"item_0","transcript":"full sentence"}`

> **Capabilities & limits**: the current realtime backend is VAD-offline — it **emits only whole-sentence `…completed`, never per-token `…delta`** (`partial_results=false`). Per-token deltas require a future vLLM streaming backend.

---

## 4. DashScope compatibility

DashScope Paraformer **recorded-file recognition** (async): submit → poll → fetch second-hop result. Client `dashscope.base_http_api_url = "http://<host>:8765/compat/dashscope/api/v1"`.

> ⚠️ The DashScope endpoint only accepts `file_urls` (a list of URLs downloaded server-side). For **local file upload**, use the OpenAI endpoint [`/audio/transcriptions`](#31-transcription-post-audiotranscriptions) (multipart) or native [`POST /v2/asr`](v2/transcription_EN.md).

### 4.1 Submit (async)

`POST /services/audio/asr/transcription`, header `X-DashScope-Async: enable` (required; missing → 400).

```json
{ "model":"paraformer-v2",
  "input":{"file_urls":["https://example.com/a.wav"]},
  "parameters":{"language_hints":["zh"],"diarization_enabled":false} }
```

| Param | Supported | Mapping |
|------|------|------|
| `input.file_urls[]` | ✅ | Server-side download (SSRF-guarded); one subtask per URL; ≤16 per request |
| `parameters.language_hints[0]` | ✅ | → recognition language |
| `parameters.diarization_enabled` | ✅ | → speaker diarization |
| `parameters.speaker_count` | ❌ ignored | Max speakers is a service-level setting (`--speaker-max`), not per-request |
| `parameters.channel_id` | ❌ | Mono, fixed channel 0 |
| `disfluency_removal_enabled` / `special_word_filter` / `timestamp_alignment_enabled` | ❌ ignored | No corresponding capability |

> ⚠️ `file_urls` must be **reachable** by this service; private/loopback addresses are blocked by default (SSRF guard), overridable with `--compat-fetch-allow-private` (trusted intranet only).

Response: `{"output":{"task_status":"PENDING","task_id":"<id>"},"request_id":"<rid>"}`

### 4.2 Poll

`POST|GET /tasks/{task_id}`:

```json
{ "output":{"task_id":"<id>","task_status":"SUCCEEDED",
    "results":[{"file_url":"https://…/a.wav",
                "transcription_url":".../tasks/<id>/transcription/0",
                "subtask_status":"SUCCEEDED"}],
    "task_metrics":{"TOTAL":1,"SUCCEEDED":1,"FAILED":0}} }
```
`task_status`: `PENDING`/`RUNNING`/`SUCCEEDED`/`FAILED` (aggregated across multiple file_urls). The result lives at `transcription_url` (second hop).

> Behind a reverse proxy / in containers, set `--compat-external-base-url` for the `transcription_url` external base; otherwise it is derived from `X-Forwarded-Proto/Host` or the request address.
> The task registry is in-memory only (with TTL): after a restart, task_ids whose results were not yet fetched return 404 and must be resubmitted.

### 4.3 Second-hop transcript

`GET /tasks/{task_id}/transcription/{idx}`, time unit **milliseconds**:

```json
{ "file_url":"https://…/a.wav",
  "transcripts":[{"channel_id":0,"content_duration_in_milliseconds":8470,"text":"full transcript",
    "sentences":[{"begin_time":0,"end_time":3200,"text":"…","sentence_id":1,"speaker_id":0,
                  "words":[{"begin_time":0,"end_time":200,"text":"hi","punctuation":""}]}]}] }
```

**Example**:
```python
import dashscope
dashscope.base_http_api_url = "http://localhost:8765/compat/dashscope/api/v1"
dashscope.api_key = "sk-xxx"
from dashscope.audio.asr import Transcription
task = Transcription.async_call(model="paraformer-v2",
        file_urls=["https://example.com/a.wav"], language_hints=["zh"])
result = Transcription.wait(task=task.output.task_id)   # SDK polls + fetches transcription_url
print(result.output)
```

### 4.4 Realtime `WS /inference`

Paraformer realtime (needs `--enable-stream`). `header/payload` envelope:

1. Client → `run-task`: `{"header":{"action":"run-task","task_id":"<uuid>","streaming":"duplex"},"payload":{"parameters":{"format":"pcm","sample_rate":16000,"language_hints":["zh"]}}}`
2. Server → `task-started`
3. Client → binary PCM frames (~100ms each)
4. Server → per sentence `result-generated`: `payload.output.sentence` with `begin_time`/`end_time`(ms)/`text`/`sentence_end:true`/`words[]`
5. Client → `finish-task` → Server → `task-finished`
6. The same connection can issue another `run-task` (connection reuse)

> **Capabilities & limits**: same as OpenAI realtime — VAD-offline emits only whole sentences (`sentence_end:true`), **no intermediate results** (`sentence_end:false`). Full intermediate results require a future vLLM streaming backend.

---

## 5. Capabilities & limits

| Capability | OpenAI compat | DashScope compat |
|------|------------|---------------|
| Offline transcription | ✅ transcriptions | ✅ recorded-file recognition |
| Local file upload | ✅ multipart | ❌ file_urls only (URLs) |
| Word-level timestamps | ✅ verbose_json + word | ✅ words[] |
| Speaker diarization | ➖ (no OpenAI field) | ✅ diarization_enabled |
| Translation | ❌ 501 | ➖ n/a |
| HTTP streaming | ✅ stream=true (SSE) | ➖ n/a |
| Realtime whole-sentence | ✅ completed (needs --enable-stream) | ✅ result-generated (needs --enable-stream) |
| Realtime per-token delta | ❌ needs vLLM | ❌ needs vLLM |
| Confidence/logprob | ➖ placeholder | ➖ not provided |
| Prompt/temperature/hotwords/disfluency | ❌ ignored | ❌ ignored |

## 6. Error codes

**OpenAI style** (`{"error":{"message","type","param","code"}}`):

| HTTP | code | Scenario |
|------|------|------|
| 400 | `invalid_value` | invalid response_format, etc. |
| 401 | `invalid_api_key` | auth failure |
| 501 | `unsupported` | translations |
| 503 | `overloaded` | task queue full |
| 504 | `timeout` | sync wait exceeded `--openai-sync-timeout` |
| 500 | `internal_error` | transcription failed |

**DashScope style** (`{"code","message","request_id"}`):

| HTTP | code | Scenario |
|------|------|------|
| 400 | `InvalidParameter` | missing X-DashScope-Async, empty/>16 file_urls |
| 401 | `InvalidApiKey` | auth failure |
| 404 | `UNKNOWN_TASK` | task_id missing/expired/lost after restart |
| subtask | `FAILED` + code | download failure/SSRF reject/transcription failure/queue busy (`Throttling`) (in `results[].subtask_status`) |

## 7. Compat vs native v2

- Need to **integrate an existing OpenAI/DashScope ecosystem** (SDKs, off-the-shelf clients) → use the compat APIs.
- Need this service's **full feature set** (voiceprint DB, task list/cancel, unified realtime envelope, per-request parameter overrides) → use native [API v2](v2_EN.md).
- Very long audio: the OpenAI sync endpoint is bounded by `--openai-sync-timeout`; for very long clips prefer the DashScope async compat or native v2 async.
