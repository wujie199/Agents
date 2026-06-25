"""L1 热记忆端口 — 提示快照 + 增量写入 + 缓存管理。"""

from typing import Protocol, Optional

from core.domain.context import RequestContext
from core.ports.memory.dtos import MemoryDelta, PromptMemorySnapshot


class HotMemoryPort(Protocol):
    """L1 热记忆契约：compose_prompt_snapshot, apply_memory_delta, update_prompt_memory, confirm_pending_deltas"""

    def compose_prompt_snapshot(
        self,
        context: RequestContext,
    ) -> PromptMemorySnapshot:
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

    async def confirm_pending_deltas(self, context: RequestContext) -> int:
        ...


class HotMemoryStore(Protocol):
    """L1 热记忆存储契约（租户 MEMORY + 用户 USER）。"""

    def compose_snapshot(self, context: RequestContext) -> PromptMemorySnapshot:
        ...

    def apply_delta(
        self,
        tenant_id: str,
        user_id: str,
        delta: MemoryDelta,
    ) -> None:
        ...

    def invalidate_cache(
        self, tenant_id: str, user_id: Optional[str] = None
    ) -> None:
        ...

    def get_snapshot_hash(
        self, tenant_id: str, user_id: str
    ) -> Optional[str]:
        ...


class HotMemoryCompressor(Protocol):
    """L1 超预算压缩契约（LLM 或截断）。"""

    async def compress_memory(self, content: str, max_chars: int) -> str:
        ...

    async def compress_user(self, content: str, max_chars: int) -> str:
        ...
