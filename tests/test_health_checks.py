"""企业级健康检查测试。"""

from __future__ import annotations

import pytest

from app.api.health_checks import run_health_checks
from app.api.rate_limit import _resolve_rate_limit_redis_url


@pytest.mark.asyncio
async def test_run_health_checks_dev():
    result = await run_health_checks(profile="dev")
    assert "status" in result
    assert "checks" in result
    assert "redis" in result["checks"]
    assert "archive" in result["checks"]


def test_rate_limit_redis_url_uses_db1(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("CHAT_RATE_LIMIT_REDIS_URL", raising=False)
    assert _resolve_rate_limit_redis_url() == "redis://localhost:6379/1"


def test_rate_limit_redis_url_explicit(monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_REDIS_URL", "redis://rl:6379/2")
    assert _resolve_rate_limit_redis_url() == "redis://rl:6379/2"
