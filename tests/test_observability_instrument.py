# -*- coding: utf-8 -*-
"""可观测性 instrument 单元测试。"""

from __future__ import annotations

import pytest

from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from core.composition.factory import FakeObservabilityPort
from core.ports.observability import Layer

from agent_platform.infrastructure.observability.adapter import (
    ObservabilityPortAdapter,
)

from app.agents.observability.instrument import record_span_metric, span_ctx


def _ctx(obs=None) -> RunContext:
    return RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="trace-abc",
            channel="test",
        ),
        observability=obs,
    )


@pytest.mark.asyncio
async def test_span_ctx_noop_without_observability():
    ctx = _ctx(obs=None)
    async with span_ctx(ctx, "test.span", Layer.AGENT) as attrs:
        attrs["ok"] = True
    # no-op, should not raise


@pytest.mark.asyncio
async def test_span_ctx_records_with_observability_adapter():
    obs = ObservabilityPortAdapter(service_name="test-instrument")
    ctx = _ctx(obs=obs)
    async with span_ctx(ctx, "test.span", Layer.AGENT, {"k": "v"}) as attrs:
        attrs["done"] = True
    spans = list(obs._spans.values())
    assert len(spans) == 1
    assert spans[0].name == "test.span"
    assert spans[0].trace_id == "trace-abc"
    assert spans[0].attributes.get("done") is True


@pytest.mark.asyncio
async def test_span_ctx_works_with_fake_observability_port():
    obs = FakeObservabilityPort()
    ctx = _ctx(obs=obs)
    async with span_ctx(ctx, "fake.span", Layer.WORKFLOW):
        pass
    assert len(obs._spans) == 1


def test_record_span_metric_noop_without_obs():
    ctx = _ctx(obs=None)
    record_span_metric(ctx, "agent.llm.duration_ms", 12.5)
    # no-op


@pytest.mark.asyncio
async def test_span_ctx_injects_identity_attributes():
    obs = ObservabilityPortAdapter(service_name="test-identity")
    ctx = _ctx(obs=obs)
    async with span_ctx(ctx, "test.span", Layer.AGENT, {"node": "prepare"}):
        pass
    span = list(obs._spans.values())[0]
    assert span.attributes.get("tenant_id") == "t1"
    assert span.attributes.get("user_id") == "u1"
    assert span.attributes.get("session_id") == "s1"
    assert span.attributes.get("node") == "prepare"


def test_record_span_metric_records_with_adapter():
    obs = ObservabilityPortAdapter(service_name="test-metric")
    ctx = _ctx(obs=obs)
    record_span_metric(ctx, "graph.node.duration_ms", 42.0, tags={"node": "prepare"})
    metrics = obs.get_metrics()
    assert "graph.node.duration_ms" in metrics
    assert metrics["graph.node.duration_ms"] == [42.0]
