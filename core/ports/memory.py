from typing import Protocol, Optional, List, Any
from dataclasses import dataclass, field

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


class MemoryPort(Protocol):
    def compose_prompt_snapshot(
        self,
        context: RequestContext,
    ) -> PromptMemorySnapshot:
        ...

    async def ensure_session(self, context: RequestContext) -> None:
        ...

    async def end_session(
        self,
        context: RequestContext,
        status: str = "closed",
        finalize: bool = True,
    ) -> None:
        ...

    async def finalize_session(self, context: RequestContext) -> None:
        ...

    async def confirm_pending_deltas(self, context: RequestContext) -> int:
        ...

    async def persist_turn(
        self,
        context: RequestContext,
        turn: TurnRecord,
    ) -> None:
        ...

    async def persist_tool_call(
        self,
        context: RequestContext,
        record: ToolCallRecord,
    ) -> None:
        ...

    async def list_turns(
        self, context: RequestContext, limit: int = 100, offset: int = 0
    ) -> List[dict]:
        ...

    async def list_sessions(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> List[dict]:
        ...

    async def apply_memory_delta(
        self,
        context: RequestContext,
        delta: MemoryDelta,
    ) -> None:
        ...

    async def update_prompt_memory(
        self,
        context: RequestContext,
        delta: MemoryDelta,
        require_hitl: bool = True,
    ) -> None:
        ...

    async def session_search(
        self,
        query: str,
        context: RequestContext,
        limit: int = 5,
        scope: str = "session",
    ) -> str:
        ...

    async def skill_search(
        self,
        query: str,
        context: RequestContext,
        limit: int = 3,
    ) -> List[SkillSummary]:
        ...

    async def run_skill(
        self,
        skill_id: str,
        inputs: dict,
        context: RequestContext,
        run_context: Any,
    ) -> Any:
        ...

    async def record_skill_outcome(
        self,
        context: RequestContext,
        outcome: SkillOutcome,
    ) -> None:
        ...

    async def resolve_entity(
        self, mention: str, context: RequestContext
    ) -> Optional[Any]:
        ...

    async def purge_user_data(
        self, tenant_id: str, user_id: str
    ) -> None:
        ...

    async def purge_expired_sessions(self, retention_days: int = 90) -> int:
        ...

    async def reindex_session_vectors(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        batch_size: int = 200,
    ) -> dict:
        ...

    async def archive_expired_sessions(
        self, retention_days: Optional[int] = None
    ) -> dict:
        ...

    async def list_cold_archives(
        self, tenant_id: str, user_id: str, limit: int = 20
    ) -> List[dict]:
        ...

    async def fetch_cold_session(self, session_id: str) -> Optional[dict]:
        ...
