"""内置 L1 Provider：始终注册，prefetch 为 no-op。"""

from __future__ import annotations

from core.domain.context import RequestContext
from core.ports.memory.provider import BaseMemoryProvider


class BuiltinProvider(BaseMemoryProvider):
    """包装现有 L1 冻结快照路径；不主动 prefetch。"""

    @property
    def name(self) -> str:
        return "builtin_l1"

    def is_available(self) -> bool:
        return True

    async def prefetch_turn(
        self, user_message: str, context: RequestContext
    ):
        return []
