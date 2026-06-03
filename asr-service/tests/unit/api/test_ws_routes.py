"""app/api/ws_routes.py 测试（TestClient websocket + 注入 fake backend）。

验证 session.created 握手、start→PCM→stop 流程、session.closed 收尾、release 释放，
以及鉴权(1008)与并发超额(1013)。
"""
import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient


class FakeSession:
    def __init__(self):
        self.configured = None

    def configure(self, msg):
        self.configured = msg

    async def feed_audio(self, data):
        yield {"type": "final", "seg_id": 0, "text": "hello", "start": 0, "end": 100}

    async def flush(self):
        yield {"type": "final", "seg_id": 1, "text": "bye", "start": 100, "end": 200}


class FakeBackend:
    mode = "standard"
    backend = "vad-offline"
    capabilities = {"partial_results": False, "word_timestamps": True, "languages_auto": True}

    def __init__(self, allow=True):
        self._allow = allow
        self.released = False

    async def acquire(self):
        return self._allow

    def create_session(self, sid):
        return FakeSession()

    def release(self, session):
        self.released = True


def _make_client(backend):
    from app.api import ws_routes
    ws_routes.init_ws_stream(backend)
    app = FastAPI()
    app.include_router(ws_routes.ws_router_stream)
    return TestClient(app)


def test_session_created_and_full_flow():
    backend = FakeBackend()
    client = _make_client(backend)
    with client.websocket_connect("/v2/asr/stream") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["mode"] == "standard"
        assert created["backend"] == "vad-offline"
        assert created["capabilities"]["word_timestamps"] is True
        assert created["protocol"] == "qwen3-asr-stream"

        ws.send_json({"type": "start", "audio_fs": 16000})
        ws.send_bytes(b"\x00\x00\x00\x00")
        final1 = ws.receive_json()
        assert final1["type"] == "final" and final1["text"] == "hello"

        ws.send_json({"type": "stop"})
        final2 = ws.receive_json()
        assert final2["type"] == "final" and final2["seg_id"] == 1
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"

    assert backend.released is True


def test_auth_rejected_without_token(monkeypatch):
    monkeypatch.setattr("app.config.API_KEY", "secret")
    client = _make_client(FakeBackend())
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/v2/asr/stream") as ws:
            ws.receive_json()
    assert exc.value.code == 1008


def test_auth_accepts_valid_token(monkeypatch):
    monkeypatch.setattr("app.config.API_KEY", "secret")
    client = _make_client(FakeBackend())
    with client.websocket_connect("/v2/asr/stream?token=secret") as ws:
        assert ws.receive_json()["type"] == "session.created"


def test_over_limit_rejected(monkeypatch):
    monkeypatch.setattr("app.config.API_KEY", "")
    client = _make_client(FakeBackend(allow=False))
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/v2/asr/stream") as ws:
            ws.receive_json()
    assert exc.value.code == 1013
