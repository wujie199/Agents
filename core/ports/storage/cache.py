from typing import Protocol, Optional, Any


class CachePort(Protocol):
    def get(self, key: str) -> Optional[Any]:
        ...

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        ...

    def delete(self, key: str) -> None:
        ...

    def expire(self, key: str, ttl_seconds: int) -> None:
        ...

    def invalidate_pattern(self, pattern: str) -> int:
        ...

    def build_key(
        self,
        tenant_id: str,
        category: str,
        identifier: str
    ) -> str:
        ...
