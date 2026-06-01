from typing import List, Optional

from core.domain.context import RequestContext
from core.ports.external_memory import Entity, Fact


class NoOpExternalMemoryAdapter:
    """L4 外部画像占位：返回空，不影响 L1 快照。"""

    async def resolve_entity(
        self, mention: str, ctx: RequestContext
    ) -> Optional[Entity]:
        return None

    async def fetch_profile_facts(
        self, user_id: str, tenant_id: str
    ) -> List[Fact]:
        return []
