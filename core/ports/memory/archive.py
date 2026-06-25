"""L2 冷档案端口 — 会话 & 对话轮次持久化。"""

from typing import Protocol, Optional, List

from core.domain.context import RequestContext
from core.ports.memory.dtos import TurnRecord, ToolCallRecord


class ArchivePort(Protocol):
    """L2 冷档案契约：会话管理 + 轮次/工具调用持久化 + 查询"""

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
