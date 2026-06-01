from typing import Protocol, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Layer(str, Enum):
    API = "api"
    WORKFLOW = "workflow"
    AGENT = "agent"
    DOMAIN = "domain"
    PLATFORM = "platform"


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    layer: Layer
    start_time: datetime
    end_time: Optional[datetime] = None
    attributes: dict = field(default_factory=dict)


class ObservabilityPort(Protocol):
    def start_span(
        self,
        trace_id: str,
        name: str,
        layer: Layer,
        parent_span_id: Optional[str] = None,
        attributes: Optional[dict] = None
    ) -> Span:
        ...

    def end_span(self, span: Span, attributes: Optional[dict] = None) -> None:
        ...

    def log_event(
        self,
        trace_id: str,
        event: str,
        layer: Layer,
        attributes: Optional[dict] = None
    ) -> None:
        ...

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[dict] = None
    ) -> None:
        ...
