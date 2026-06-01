from typing import Optional, Any, List
from datetime import datetime
from enum import Enum
import uuid
import logging
from core.ports.observability import Layer, Span


class ObservabilityPortAdapter:
    def __init__(self, service_name: str = "agents"):
        self._service_name = service_name
        self._spans: dict[str, Span] = {}
        self._active_spans: dict[str, Span] = {}
        self._metrics: dict[str, List[float]] = {}
        self._logger = logging.getLogger(service_name)
    
    def start_span(
        self,
        trace_id: str,
        name: str,
        layer: Layer,
        parent_span_id: Optional[str] = None,
        attributes: Optional[dict] = None
    ) -> Span:
        span_id = str(uuid.uuid4())[:8]
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            layer=layer,
            start_time=datetime.now(),
            attributes=attributes or {}
        )
        
        self._spans[span_id] = span
        self._active_spans[span_id] = span
        
        self._logger.debug(
            f"[SPAN START] trace={trace_id} span={span_id} name={name} layer={layer.value}"
        )
        
        return span
    
    def end_span(
        self,
        span: Span,
        attributes: Optional[dict] = None
    ) -> None:
        span.end_time = datetime.now()
        
        if attributes:
            span.attributes.update(attributes)
        
        if span.span_id in self._active_spans:
            del self._active_spans[span.span_id]
        
        duration_ms = 0
        if span.end_time and span.start_time:
            duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        
        self._logger.debug(
            f"[SPAN END] trace={span.trace_id} span={span.span_id} "
            f"name={span.name} duration_ms={duration_ms:.2f}"
        )
    
    def log_event(
        self,
        trace_id: str,
        event: str,
        layer: Layer,
        attributes: Optional[dict] = None
    ) -> None:
        attrs_str = ""
        if attributes:
            attrs_str = " " + " ".join(f"{k}={v}" for k, v in attributes.items())
        
        self._logger.info(
            f"[EVENT] trace={trace_id} event={event} layer={layer.value}{attrs_str}"
        )
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[dict] = None
    ) -> None:
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)
        
        tags_str = ""
        if tags:
            tags_str = " " + " ".join(f"{k}={v}" for k, v in tags.items())
        
        self._logger.debug(f"[METRIC] {name}={value}{tags_str}")
    
    def get_span(self, span_id: str) -> Optional[Span]:
        return self._spans.get(span_id)
    
    def get_active_spans(self) -> List[Span]:
        return list(self._active_spans.values())
    
    def get_metrics(self) -> dict[str, List[float]]:
        return self._metrics.copy()
    
    def get_metric_stats(self, name: str) -> Optional[dict]:
        values = self._metrics.get(name, [])
        if not values:
            return None
        
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    
    def clear(self) -> None:
        self._spans.clear()
        self._active_spans.clear()
        self._metrics.clear()
