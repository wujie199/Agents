"""聊天 RunContext 工厂测试。"""

from __future__ import annotations

import os
from unittest.mock import patch

from core.domain.context import RequestContext

from app.agents.context_factory import build_chat_run_context
from core.composition.production_factory import build_cache_port


def _request() -> RequestContext:
    return RequestContext(
        tenant_id="t",
        user_id="u",
        session_id="s",
        trace_id="tr",
        channel="test",
    )


def test_build_chat_run_context_dev(monkeypatch):
    monkeypatch.setenv("RAG_LEGACY_DEFAULT", "false")
    with patch(
        "app.agents.context_factory.build_development_context"
    ) as mock_dev:
        from core.composition.run_context import RunContext

        mock_dev.return_value = RunContext(request=_request(), extra={})
        ctx = build_chat_run_context(_request(), profile="dev", data_dir="data")
        mock_dev.assert_called_once()
        assert ctx.extra.get("rag_tenant_id") == "t"
        assert "rag_chroma_dir" in ctx.extra


def test_build_chat_run_context_production():
    with patch(
        "app.agents.context_factory.build_production_context"
    ) as mock_prod:
        from core.composition.run_context import RunContext

        mock_prod.return_value = RunContext(request=_request(), extra={})
        ctx = build_chat_run_context(
            _request(), profile="production", data_dir="data"
        )
        mock_prod.assert_called_once()
        assert ctx.extra.get("rag_tenant_id") == "t"
        assert ctx.extra.get("rag_chroma_dir") == "data/chroma"


def test_build_cache_port_uses_redis_url(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://redis.example:6380/2")
    cache = build_cache_port()
    assert cache.__class__.__name__ == "EnterpriseRedisCacheAdapter"
    assert cache._pool.connection_kwargs["host"] == "redis.example"
    assert cache._pool.connection_kwargs["port"] == 6380
    assert cache._pool.connection_kwargs["db"] == 2
