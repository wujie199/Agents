from typing import Protocol, Optional, List
from dataclasses import dataclass

from core.domain.context import RequestContext
from core.domain.evidence import EvidenceBundle


@dataclass
class RetrieveRequest:
    query: str
    context_key: Optional[str] = None
    plan_override: Optional[dict] = None


class RAGPort(Protocol):
    async def route_and_retrieve(
        self,
        query: str,
        context: RequestContext,
        plan: Optional[dict] = None,
    ) -> EvidenceBundle:
        ...

    async def route_and_retrieve_batch(
        self,
        requests: List[RetrieveRequest],
        context: RequestContext,
        plan: Optional[dict] = None,
    ) -> List[EvidenceBundle]:
        ...

    async def invalidate_document(
        self,
        doc_id: str,
        tenant_id: str,
    ) -> None:
        ...

    async def health(self) -> dict:
        ...
