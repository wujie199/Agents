from typing import Protocol
from core.domain.context import ACL


class IdentityPort(Protocol):
    def resolve_acl(
        self,
        tenant_id: str,
        user_id: str,
        channel: str
    ) -> ACL:
        ...

    def validate_tenant(self, tenant_id: str) -> bool:
        ...

    def validate_user(self, tenant_id: str, user_id: str) -> bool:
        ...
