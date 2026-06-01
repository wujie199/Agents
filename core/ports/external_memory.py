from typing import Protocol, Optional, List
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


class ExternalMemoryProvider(Protocol):
    async def resolve_entity(
        self, mention: str, ctx: RequestContext
    ) -> Optional[Entity]:
        ...

    async def fetch_profile_facts(
        self, user_id: str, tenant_id: str
    ) -> List[Fact]:
        ...
