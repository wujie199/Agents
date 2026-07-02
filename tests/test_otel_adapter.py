# -*- coding: utf-8 -*-
"""OpenTelemetry adapter 单元测试。"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agent_platform.infrastructure.observability.adapter import (
    ObservabilityPortAdapter,
)
from agent_platform.infrastructure.observability.otel_adapter import (
    OtelObservabilityAdapter,
    build_observability_port,
    try_create_otel_adapter,
)
from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from core.ports.observability import Layer


def test_build_observability_port_no_endpoint():
    with patch.dict(os.environ, {}, clear=True):
        obs = build_observability_port(service_name="test-svc")
    assert isinstance(obs, ObservabilityPortAdapter)
    assert not isinstance(obs, OtelObservabilityAdapter) or not obs.otel_enabled


def test_build_observability_port_endpoint_without_packages():
    env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318/v1/traces",
        "OTEL_SERVICE_NAME": "agents-chat",
    }
    with patch.dict(os.environ, env, clear=True):
        obs = build_observability_port(service_name="fallback")
    assert isinstance(obs, ObservabilityPortAdapter)
    assert obs._service_name == "agents-chat"


def test_otel_adapter_fail_open_on_missing_packages():
    with patch(
        "agent_platform.infrastructure.observability.otel_adapter._load_otel_modules",
        return_value=False,
    ):
        adapter = OtelObservabilityAdapter(
            "svc",
            "http://localhost:4318/v1/traces",
            sample_rate=1.0,
        )
    assert adapter.otel_enabled is False
    span = adapter.start_span("tr", "test", Layer.AGENT, attributes={"node": "agent"})
    adapter.end_span(span, attributes={"duration_ms": 1.0})
    assert span.span_id in adapter._spans


def test_try_create_otel_adapter_with_mock_otel():
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_span.return_value = mock_span
    fake_otel = {
        "trace": MagicMock(),
        "OTLPSpanExporter": MagicMock(),
        "Resource": MagicMock(),
        "TracerProvider": MagicMock(),
        "BatchSpanProcessor": MagicMock(),
        "TraceIdRatioBased": MagicMock(),
        "set_span_in_context": MagicMock(return_value=None),
    }
    fake_otel["trace"].get_tracer.return_value = mock_tracer

    with patch(
        "agent_platform.infrastructure.observability.otel_adapter._load_otel_modules",
        return_value=fake_otel,
    ):
        adapter = try_create_otel_adapter(
            "agents-chat",
            "http://collector:4318/v1/traces",
            sample_rate=0.1,
        )

    assert adapter is not None
    assert adapter.otel_enabled is True
    span = adapter.start_span(
        "trace-1",
        "graph.agent",
        Layer.AGENT,
        attributes={"tenant_id": "t1", "node": "agent", "tool_name": "memory"},
    )
    adapter.end_span(span, attributes={"duration_ms": 12.5})
    mock_tracer.start_span.assert_called_once()
    mock_span.end.assert_called_once()


def test_otel_adapter_records_metrics_in_memory():
    mock_tracer = MagicMock()
    fake_otel = {
        "trace": MagicMock(),
        "OTLPSpanExporter": MagicMock(),
        "Resource": MagicMock(),
        "TracerProvider": MagicMock(),
        "BatchSpanProcessor": MagicMock(),
        "TraceIdRatioBased": MagicMock(),
        "set_span_in_context": MagicMock(return_value=None),
    }
    fake_otel["trace"].get_tracer.return_value = mock_tracer

    with patch(
        "agent_platform.infrastructure.observability.otel_adapter._load_otel_modules",
        return_value=fake_otel,
    ):
        adapter = try_create_otel_adapter("svc", "http://localhost:4318/v1/traces")

    assert adapter is not None
    adapter.record_metric("agent.tool.calls_total", 1.0, {"tool_name": "memory"})
    assert "agent.tool.calls_total" in adapter.get_metrics()
