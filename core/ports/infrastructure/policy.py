from typing import Protocol, Optional, Any
from dataclasses import dataclass


@dataclass
class PolicyResult:
    allowed: bool
    reason: Optional[str] = None
    suggested_value: Optional[Any] = None


class PolicyPort(Protocol):
    def check_rate_limit(
        self,
        tenant_id: str,
        user_id: str,
        action: str
    ) -> PolicyResult:
        ...

    def check_token_budget(
        self,
        tenant_id: str,
        user_id: str,
        requested_tokens: int
    ) -> PolicyResult:
        ...

    def get_max_parallel_sends(self, tenant_id: str) -> int:
        ...

    def get_max_intra_batch_workers(self, tenant_id: str) -> int:
        ...

    def get_batch_size(
        self,
        tenant_id: str,
        context: Optional[dict] = None
    ) -> int:
        ...

    def suggest_batch_size(
        self,
        tenant_id: str,
        item_count: int,
        context: Optional[dict] = None
    ) -> int:
        ...
