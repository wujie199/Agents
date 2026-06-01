from typing import Protocol, Optional, List
from dataclasses import dataclass
from core.domain.context import RequestContext


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
class MemoryDelta:
    key: str
    value: str
    source: str


@dataclass
class SkillSummary:
    skill_id: str
    title: str
    summary: str


class MemoryPort(Protocol):
    def compose_prompt_snapshot(
        self,
        context: RequestContext
    ) -> PromptMemorySnapshot:
        ...

    async def persist_turn(
        self,
        context: RequestContext,
        turn: TurnRecord
    ) -> None:
        ...

    async def update_prompt_memory(
        self,
        context: RequestContext,
        delta: MemoryDelta,
        require_hitl: bool = True
    ) -> None:
        ...

    async def session_search(
        self,
        query: str,
        context: RequestContext,
        limit: int = 5
    ) -> str:
        ...

    async def skill_search(
        self,
        query: str,
        context: RequestContext,
        limit: int = 3
    ) -> List[SkillSummary]:
        ...
