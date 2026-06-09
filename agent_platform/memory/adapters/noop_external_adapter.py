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

    async def list_profile_users(self, tenant_id: str) -> List[str]:
        return []

    async def get_profile(self, tenant_id: str, user_id: str) -> dict:
        return {}

    async def save_profile(
        self, tenant_id: str, user_id: str, profile: dict
    ) -> None:
        return None

    async def upsert_profile_facts(
        self, tenant_id: str, user_id: str, facts: List[dict]
    ) -> int:
        return 0

    async def delete_profile(self, tenant_id: str, user_id: str) -> bool:
        return False

    async def purge_tenant_profiles(self, tenant_id: str) -> int:
        return 0
