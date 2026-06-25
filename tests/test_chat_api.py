"""HTTP Chat API 单元测试（需 fastapi）。"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.api.chat_server import create_app
from app.agents.orchestration.chat_turn import ChatTurnResult


@pytest.fixture
def client():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CHAT_API_KEY", None)
        app = create_app(config_dir="config", data_dir="data")
        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["profile"] == "dev"


def test_chat_turn_and_end(client):
    mock_result = ChatTurnResult(
        assistant_text="API 回复",
        evidence_count=1,
        rag_empty=False,
        history_turns=2,
    )

    with patch(
        "app.api.chat_server.execute_chat_turn",
        new=AsyncMock(return_value=mock_result),
    ):
        r = client.post(
            "/v1/chat/turn",
            json={
                "tenant_id": "t1",
                "user_id": "u1",
                "session_id": "api1",
                "message": "你好世界",
                "engine": "langgraph",
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["assistant_text"] == "API 回复"
    assert body["session_id"] == "api1"

    with patch(
        "app.api.chat_server.end_agent_session",
        new=AsyncMock(),
    ):
        r2 = client.post(
            "/v1/chat/end",
            json={
                "tenant_id": "t1",
                "user_id": "u1",
                "session_id": "api1",
            },
        )
    assert r2.status_code == 200
    assert r2.json()["ok"] is True


def test_chat_turn_stream(client):
    mock_result = ChatTurnResult(
        assistant_text="流式回复内容",
        evidence_count=0,
        rag_empty=True,
        history_turns=0,
    )

    async def _fake_stream(*_a, **_k):
        yield json.dumps({"type": "meta", "evidence_count": 0}, ensure_ascii=False)
        yield json.dumps({"type": "delta", "text": "流"}, ensure_ascii=False)
        yield json.dumps(
            {"type": "done", "assistant_text": "流式回复内容"}, ensure_ascii=False
        )

    with patch(
        "app.api.chat_server.stream_chat_turn_events",
        new=_fake_stream,
    ):
        with client.stream(
            "POST",
            "/v1/chat/turn/stream",
            json={
                "tenant_id": "t1",
                "user_id": "u1",
                "session_id": "stream1",
                "message": "test",
            },
        ) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
    assert "delta" in body
    assert "done" in body


def test_api_key_required_when_configured(client):
    mock_result = ChatTurnResult(assistant_text="x")
    with patch.dict(os.environ, {"CHAT_API_KEY": "secret"}):
        app = create_app(config_dir="config", data_dir="data")
        with TestClient(app) as authed_client:
            with patch(
                "app.api.chat_server.execute_chat_turn",
                new=AsyncMock(return_value=mock_result),
            ):
                denied = authed_client.post(
                    "/v1/chat/turn",
                    json={
                        "tenant_id": "t1",
                        "user_id": "u1",
                        "session_id": "s1",
                        "message": "hi",
                    },
                )
                assert denied.status_code == 401

                ok = authed_client.post(
                    "/v1/chat/turn",
                    headers={"Authorization": "Bearer secret"},
                    json={
                        "tenant_id": "t1",
                        "user_id": "u1",
                        "session_id": "s1",
                        "message": "hi",
                    },
                )
                assert ok.status_code == 200


def test_rate_limit_returns_429(client):
    mock_result = ChatTurnResult(assistant_text="x")
    with patch(
        "app.api.chat_server.create_chat_rate_limiter"
    ) as mock_cls:
        inst = mock_cls.return_value
        inst.check.return_value = (False, "too many")
        app = create_app(config_dir="config", data_dir="data")
        with TestClient(app) as limited:
            with patch(
                "app.api.chat_server.execute_chat_turn",
                new=AsyncMock(return_value=mock_result),
            ):
                r = limited.post(
                    "/v1/chat/turn",
                    json={
                        "tenant_id": "t1",
                        "user_id": "u1",
                        "session_id": "rl1",
                        "message": "hi",
                    },
                )
            assert r.status_code == 429


def test_chat_end_unknown_session(client):
    r = client.post(
        "/v1/chat/end",
        json={
            "tenant_id": "t1",
            "user_id": "u1",
            "session_id": "missing",
        },
    )
    assert r.status_code == 404
