"""企业记忆 API 单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.api.chat_server import create_app


@pytest.fixture
def client():
    app = create_app(config_dir="config", data_dir="data")
    with TestClient(app) as c:
        yield c


def test_memory_config_endpoint(client):
    r = client.get("/v1/memory/config")
    assert r.status_code == 200
    assert "archive_backend" in r.json()


def test_memory_status_endpoint(client):
    mock_status = {
        "tenant_id": "t1",
        "user_id": "u1",
        "l1_hash": "h",
        "l1_chars": 10,
        "pending_l1_count": 0,
        "recent_sessions": [],
        "config": {},
        "metrics": {},
    }
    with patch(
        "app.api.memory_routes.get_memory_status",
        new=AsyncMock(return_value=mock_status),
    ):
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
