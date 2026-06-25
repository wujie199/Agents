"""L4 外部画像端口 — 实体解析 + Profile 读写。"""

from typing import Protocol, Optional, List, Any
from dataclasses import dataclass

from core.domain.context import RequestContext


@dataclass
class Entity:
    mention: str
    canonical_id: str
    display_name: str


@dataclass
class Fact:
    key: str
    value: str
    source: str


class ExternalProfilePort(Protocol):
    """L4 外部画像契约：实体解析 + Profile 管理"""

    async def resolve_entity(
        self, mention: str, context: RequestContext
    ) -> Optional[Any]:
        ...

    async def fetch_profile_facts(
        self, tenant_id: str, user_id: str
    ) -> List[dict]:
        ...

    async def list_profile_users(self, tenant_id: str) -> List[str]:
        ...

    async def get_profile(self, tenant_id: str, user_id: str) -> dict:
        ...

    async def import_profile(
        self, tenant_id: str, user_id: str, profile: dict
    ) -> None:
        ...

    async def set_profile_facts(
        self, tenant_id: str, user_id: str, facts: List[dict]
    ) -> int:
        ...

    async def purge_tenant_l4(self, tenant_id: str) -> dict:
        ...


class ExternalMemoryProvider(Protocol):
    """L4 外部记忆提供方底层契约。"""

    async def resolve_entity(
        self, mention: str, ctx: RequestContext
    ) -> Optional[Entity]:
        ...

    async def fetch_profile_facts(
        self, user_id: str, tenant_id: str
    ) -> List[Fact]:
        ...

    async def list_profile_users(self, tenant_id: str) -> List[str]:
        ...

    async def get_profile(self, tenant_id: str, user_id: str) -> dict:
        ...

    async def save_profile(
        self, tenant_id: str, user_id: str, profile: dict
    ) -> None:
        ...

    async def upsert_profile_facts(
        self, tenant_id: str, user_id: str, facts: List[dict]
    ) -> int:
        ...

    async def delete_profile(self, tenant_id: str, user_id: str) -> bool:
        ...

    async def purge_tenant_profiles(self, tenant_id: str) -> int:
        ...
