"""企业记忆 API 单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from core.domain.context import RequestContext
from core.composition.run_context import RunContext

from app.api.chat_server import create_app
from app.agents.orchestration.chat_config import load_chat_config
from app.agents.orchestration.chat_service import ChatSessionHandle


def _light_run_ctx() -> RunContext:
    memory = MagicMock()
    memory.compose_prompt_snapshot.return_value = MagicMock(
        hash="h", memory_text="test"
    )
    memory.list_sessions = AsyncMock(return_value=[])
    return RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
    )


@pytest.fixture
def client():
    handle = ChatSessionHandle(
        run_ctx=_light_run_ctx(),
        chat_cfg=load_chat_config("config", profile="dev"),
    )
    with patch(
        "app.api.chat_server.bootstrap_memory_runtime",
        new=AsyncMock(return_value={"memory_ready": True}),
    ), patch(
        "app.api.chat_server.build_chat_run_context",
        return_value=_light_run_ctx(),
    ), patch(
        "app.api.chat_server.ChatSessionRegistry.get_or_create",
        new=AsyncMock(return_value=handle),
    ):
        app = create_app(config_dir="config", data_dir="data")
        with TestClient(app) as c:
            yield c


def test_memory_config_endpoint(client):
    r = client.get("/v1/memory/config")
    assert r.status_code == 200
    assert "archive_backend" in r.json()


def test_memory_status_endpoint(client):
    r = client.post(
        "/v1/memory/status",
        json={"tenant_id": "t1", "user_id": "u1", "session_id": "s1"},
    )
    assert r.status_code == 200
    assert r.json()["l1_hash"] == "h"


def test_metrics_endpoints(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")

    r2 = client.get("/v1/metrics/memory")
    assert r2.status_code == 200
    assert "metrics" in r2.json()
