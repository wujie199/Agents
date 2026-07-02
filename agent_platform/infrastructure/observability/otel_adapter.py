# -*- coding: utf-8 -*-
"""OpenTelemetry 桥接：在 ObservabilityPortAdapter 之上导出 OTLP trace（fail-open）。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from core.ports.observability import Layer, Span

from agent_platform.infrastructure.observability.adapter import ObservabilityPortAdapter

logger = logging.getLogger(__name__)

_OTEL_MODULES: Any = None


def _load_otel_modules() -> Any:
    """惰性加载 OTel SDK；未安装时返回 False。"""
    global _OTEL_MODULES
    if _OTEL_MODULES is not None:
        return _OTEL_MODULES
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        from opentelemetry.trace import set_span_in_context

        _OTEL_MODULES = {
            "trace": trace,
            "OTLPSpanExporter": OTLPSpanExporter,
            "Resource": Resource,
            "TracerProvider": TracerProvider,
            "BatchSpanProcessor": BatchSpanProcessor,
            "TraceIdRatioBased": TraceIdRatioBased,
            "set_span_in_context": set_span_in_context,
        }
    except ImportError:
        _OTEL_MODULES = False
    return _OTEL_MODULES


def _parse_sample_rate(raw: Optional[str] = None) -> float:
    value = raw if raw is not None else os.environ.get("OTEL_TRACES_SAMPLER_ARG", "1.0")
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, rate))


def _otel_attributes(attributes: Optional[dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, val in (attributes or {}).items():
        if val is None:
            continue
        out[str(key)] = str(val)
    return out


class OtelObservabilityAdapter(ObservabilityPortAdapter):
    """双写：内存 metrics + OTLP span 导出。"""

    def __init__(
        self,
        service_name: str,
        endpoint: str,
        *,
        sample_rate: float = 1.0,
    ) -> None:
        super().__init__(service_name=service_name)
        self._otel_spans: dict[str, Any] = {}
        self._tracer = None
        otel = _load_otel_modules()
        if otel is False:
            logger.warning(
                "OTEL_EXPORTER_OTLP_ENDPOINT 已设置但 OpenTelemetry 包未安装，"
                "回退内存 ObservabilityPortAdapter"
            )
            return
        try:
            resource = otel["Resource"].create({"service.name": service_name})
            sampler = otel["TraceIdRatioBased"](float(sample_rate))
            provider = otel["TracerProvider"](resource=resource, sampler=sampler)
            exporter = otel["OTLPSpanExporter"](endpoint=endpoint)
            provider.add_span_processor(otel["BatchSpanProcessor"](exporter))
            otel["trace"].set_tracer_provider(provider)
            self._tracer = otel["trace"].get_tracer(service_name)
            self._otel = otel
        except Exception as exc:
            logger.warning("OpenTelemetry 初始化失败，回退内存 adapter: %s", exc)
            self._tracer = None

    @property
    def otel_enabled(self) -> bool:
        return self._tracer is not None

    def start_span(
        self,
        trace_id: str,
        name: str,
        layer: Layer,
        parent_span_id: Optional[str] = None,
        attributes: Optional[dict] = None,
    ) -> Span:
        span = super().start_span(
            trace_id,
            name,
            layer,
            parent_span_id=parent_span_id,
            attributes=attributes,
        )
        if self._tracer is None:
            return span

        attrs = _otel_attributes(attributes)
        attrs.setdefault("layer", layer.value)
        attrs.setdefault("trace_id", trace_id)

        parent_ctx = None
        if parent_span_id:
            parent_otel = self._otel_spans.get(parent_span_id)
            if parent_otel is not None:
                parent_ctx = self._otel["set_span_in_context"](parent_otel)

        otel_span = self._tracer.start_span(name, context=parent_ctx, attributes=attrs)
        self._otel_spans[span.span_id] = otel_span
        return span

    def end_span(self, span: Span, attributes: Optional[dict] = None) -> None:
        super().end_span(span, attributes=attributes)
        otel_span = self._otel_spans.pop(span.span_id, None)
        if otel_span is None:
            return
        for key, val in _otel_attributes(attributes).items():
            otel_span.set_attribute(key, val)
        otel_span.end()


def try_create_otel_adapter(
    service_name: str,
    endpoint: str,
    *,
    sample_rate: Optional[float] = None,
) -> Optional[OtelObservabilityAdapter]:
    """创建 OTel adapter；包缺失或初始化失败时返回 None。"""
    rate = _parse_sample_rate(str(sample_rate)) if sample_rate is not None else _parse_sample_rate()
    adapter = OtelObservabilityAdapter(
        service_name=service_name,
        endpoint=endpoint,
        sample_rate=rate,
    )
    if not adapter.otel_enabled:
        return None
    return adapter


def build_observability_port(service_name: str = "agents") -> ObservabilityPortAdapter:
    """按环境变量选择 ObservabilityPortAdapter 或 OtelObservabilityAdapter。"""
    endpoint = (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    svc = (os.environ.get("OTEL_SERVICE_NAME") or service_name).strip() or service_name
    if not endpoint:
        return ObservabilityPortAdapter(service_name=svc)
    adapter = try_create_otel_adapter(svc, endpoint)
    if adapter is not None:
        return adapter
    return ObservabilityPortAdapter(service_name=svc)
