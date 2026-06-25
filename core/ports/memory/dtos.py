"""Memory DTOs — data objects shared across memory sub-protocols."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PromptMemorySnapshot:
    memory_text: str
    hash: str
    frozen: bool = True


@dataclass
class TurnRecord:
    role: str
    content: str
    tool_calls: Optional[list] = None
    ts: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class ToolCallRecord:
    tool_name: str
    args_hash: Optional[str] = None
    result_summary: Optional[str] = None
    status: str = "ok"
    latency_ms: Optional[int] = None
    ts: Optional[str] = None


@dataclass
class MemoryDelta:
    key: str
    value: str
    source: str


@dataclass
class SkillSummary:
    skill_id: str
    title: str
    summary: str
    success_rate: float = 1.0
    last_used_at: Optional[str] = None
    usage_count: int = 0
    anti_patterns: List[str] = field(default_factory=list)
    status: str = "active"


@dataclass
class SkillOutcome:
    skill_id: str
    success: bool
    steps_executed: int = 0
    error: Optional[str] = None


@dataclass
class SessionFragment:
    message_id: str
    session_id: str
    role: str
    content: str
    ts: str
    score: float = 0.0
    source: str = "online"


@dataclass
class SessionSearchResult:
    summary: str
    fragments: List[SessionFragment] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "sources": self.sources,
            "fragments": [
                {
                    "message_id": f.message_id,
                    "session_id": f.session_id,
                    "role": f.role,
                    "content": f.content,
                    "ts": f.ts,
                    "score": f.score,
                    "source": f.source,
                }
                for f in self.fragments
            ],
        }
