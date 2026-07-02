# -*- coding: utf-8 -*-
"""Phase 4 可观测性：审计持久化、错误分类、Redis 限流回退。"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_platform.infrastructure.observability.adapter import ObservabilityPortAdapter
from app.agents.middleware.audit import AuditMiddleware
from app.agents.middleware.error_classifier import (
    ErrorClassifierMiddleware,
    classify_exception,
)
from app.agents.middleware.policy import PolicyMiddleware
from app.agents.middleware.rate_limit_backend import (
    InMemoryQpsBackend,
    RedisQpsBackend,
    create_qps_backend,
)
from core.composition.run_context import RunContext
from core.domain.context import RequestContext


class _RunCtx:
    def __init__(self, tenant_id: str = "t1", obs=None):
        self.request = type(
            "R",
            (),
            {
                "tenant_id": tenant_id,
                "user_id": "u1",
                "session_id": "s1",
            },
        )()
        self.observability = obs


@pytest.mark.asyncio
async def test_audit_persist_writes_ndjson(tmp_path):
    mw = AuditMiddleware(persist=True, audit_log_dir=str(tmp_path))
    result = {"assistant_text": "hello", "user_input": "hi"}
    config = {"configurable": {"run_ctx": _RunCtx()}}
    await mw.on_exit(
        "agent",
        {},
        config,
        result,
        extra={"trace_id": "tr-1", "span_id": "sp-1", "duration_ms": 12.5},
    )
    files = list(tmp_path.glob("audit_*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert row["trace_id"] == "tr-1"
    assert row["tenant_id"] == "t1"
    assert row["node"] == "agent"
    assert "assistant_text" in row["content_hashes"]


@pytest.mark.parametrize(
    "exc,expected",
    [
        (PermissionError("denied"), "policy_denied"),
        (TimeoutError("llm timeout"), "llm_timeout"),
        (asyncio.TimeoutError(), "llm_timeout"),
        (RuntimeError("tool execution failed"), "tool_error"),
        (RuntimeError("memory archive unavailable"), "memory_error"),
        (ValueError("boom"), "unknown"),
    ],
)
def test_classify_exception(exc, expected):
    assert classify_exception(exc) == expected


@pytest.mark.asyncio
async def test_error_classifier_records_metric():
    obs = ObservabilityPortAdapter(service_name="phase4")
    run_ctx = RunContext(
        request=RequestContext(
            tenant_id="t-err",
            user_id="u",
            session_id="s",
            trace_id="tr",
            channel="test",
        ),
        observability=obs,
    )
    mw = ErrorClassifierMiddleware()
    config = {"configurable": {"run_ctx": run_ctx}}
    await mw.on_exit(
        "prepare",
        {},
        config,
        None,
        error=PermissionError("rate limit"),
        extra={},
    )
    metrics = obs.get_metrics()
    assert "graph.node.errors_total" in metrics
    assert metrics["graph.node.errors_total"] == [1.0]


def test_redis_qps_fallback_to_memory():
    broken = MagicMock()
    broken.pipeline.side_effect = ConnectionError("redis down")
    fallback = InMemoryQpsBackend()
    backend = RedisQpsBackend(broken, fallback=fallback)
    assert backend.allow("tenant-a", max_qps=1) is True
    assert backend.allow("tenant-a", max_qps=1) is False
    assert broken.pipeline.called


def test_create_qps_backend_memory_when_no_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    backend = create_qps_backend(backend="memory")
    assert isinstance(backend, InMemoryQpsBackend)


def test_create_qps_backend_redis_fail_open(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:9/0")
    backend = create_qps_backend(backend="redis", redis_url=os.environ["REDIS_URL"])
    assert isinstance(backend, InMemoryQpsBackend)


@pytest.mark.asyncio
async def test_policy_middleware_uses_injected_backend():
    backend = InMemoryQpsBackend()
    mw = PolicyMiddleware(max_qps_per_tenant=1, qps_backend=backend)
    config = {"configurable": {"run_ctx": _RunCtx("tenant-x")}}
    await mw.on_enter("n", {}, config)
    with pytest.raises(PermissionError):
        await mw.on_enter("n", {}, config)
