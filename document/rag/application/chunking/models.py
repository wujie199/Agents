"""七步分块流水线内部数据结构。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StructuralUnit:
    unit_type: str
    content: str
    heading_path: str = ""
    position: int = 0
    is_fragment: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentenceSpan:
    text: str
    unit_index: int
    global_index: int
    char_start: int = 0
    char_end: int = 0


@dataclass
class CutPoint:
    sentence_index: int
    confidence: str  # confirmed | weak_A | weak_B
    reason: str = ""


@dataclass
class ForbiddenRange:
    start: int
    end: int
    forbidden_type: str


@dataclass
class BoundaryResult:
    confirmed: List[CutPoint] = field(default_factory=list)
    weak_a: List[CutPoint] = field(default_factory=list)
    weak_b: List[CutPoint] = field(default_factory=list)
    forbidden: List[ForbiddenRange] = field(default_factory=list)


@dataclass
class ScoredChunk:
    content: str
    unit_type: str = "paragraph"
    heading_path: str = ""
    position: int = 0
    density: float = 0.5
    entities: List[str] = field(default_factory=list)
    size: int = 0
    score: float = 1.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    contextualized_content: Optional[str] = None
    inherited_entities: Dict[str, str] = field(default_factory=dict)
    quality: str = "high"
    chunk_role: str = "retrieval"  # retrieval | parent | storage


@dataclass
class RepairTask:
    task_type: str  # reference | entity | context | structure
    chunk_index: int
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    retrieval_chunks: List[ScoredChunk] = field(default_factory=list)
    parent_chunks: List[ScoredChunk] = field(default_factory=list)
    repair_tasks: List[RepairTask] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
