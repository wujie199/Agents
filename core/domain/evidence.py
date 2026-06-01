from dataclasses import dataclass, field
from typing import Optional, List, Literal
from enum import Enum


class SourceType(str, Enum):
    VECTOR = "vector"
    SQL = "sql"
    GRAPH = "graph"
    CACHE = "cache"


class DegradedReason(str, Enum):
    ALL_BACKENDS_FAILED = "all_backends_failed"
    VECTOR_UNAVAILABLE = "vector_unavailable"
    GRAPH_UNAVAILABLE = "graph_unavailable"
    PARTIAL_RESULTS = "partial_results"


@dataclass
class Evidence:
    id: str
    content: str
    source_type: SourceType
    score: float = 0.0
    citation: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    evidences: List[Evidence] = field(default_factory=list)
    plan: Optional[dict] = None
    empty: bool = False
    degraded_reason: Optional[DegradedReason] = None
    error_code: Optional[str] = None

    def __post_init__(self):
        if not self.evidences and not self.empty:
            self.empty = True

    @classmethod
    def empty_bundle(
        cls,
        reason: DegradedReason,
        error_code: str,
        plan: Optional[dict] = None
    ) -> "EvidenceBundle":
        return cls(
            evidences=[],
            empty=True,
            degraded_reason=reason,
            error_code=error_code,
            plan=plan
        )

    def is_degraded(self) -> bool:
        return self.degraded_reason is not None

    def total_content_length(self) -> int:
        return sum(len(e.content) for e in self.evidences)
