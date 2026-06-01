from typing import Protocol, Optional

from core.ports.memory import MemoryDelta, PromptMemorySnapshot
from core.domain.context import RequestContext


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
