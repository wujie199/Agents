"""并行 Worker 批量刷 L2 archive。"""

from __future__ import annotations

from typing import Any, List, Tuple

from core.domain.context import RequestContext
from core.ports.memory import TurnRecord


class TurnBuffer:
    def __init__(self, memory: Any, *, flush_size: int = 10):
        self._memory = memory
        self._flush_size = max(1, flush_size)
        self._pending: List[Tuple[RequestContext, TurnRecord]] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def append(self, context: RequestContext, turn: TurnRecord) -> None:
        self._pending.append((context, turn))
        if len(self._pending) >= self._flush_size:
            await self.flush()

    async def flush(self) -> int:
        batch = self._pending
        self._pending = []
        for context, turn in batch:
            await self._memory.persist_turn(context, turn)
        return len(batch)

    def pending_turns_for(self, context: RequestContext) -> List[dict]:
        """Buffer 中尚未 flush 的 turns（供对话历史即时可见）。"""
        rows: List[dict] = []
        for ctx, turn in self._pending:
            if (
                ctx.tenant_id == context.tenant_id
                and ctx.user_id == context.user_id
                and ctx.session_id == context.session_id
            ):
                rows.append(
                    {
                        "role": turn.role,
                        "content": turn.content,
                        "trace_id": turn.trace_id,
                    }
                )
        return rows
