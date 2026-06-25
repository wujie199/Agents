"""记忆运维端口 — 数据清理、重建索引、归档管理。"""

from typing import Protocol, Optional, List

from core.domain.context import RequestContext


class MemoryAdminPort(Protocol):
    """记忆运维契约：数据清理、重建索引、归档管理"""

    async def purge_user_data(
        self, tenant_id: str, user_id: str
    ) -> dict:
        ...

    async def backfill_cold_search_index(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict:
        ...

    async def purge_expired_sessions(self, retention_days: int = 90) -> int:
        ...

    async def reindex_session_vectors(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        batch_size: int = 200,
    ) -> dict:
        ...

    async def archive_expired_sessions(
        self, retention_days: Optional[int] = None
    ) -> dict:
        ...

    async def archive_session(self, session_id: str) -> dict:
        ...

    async def list_cold_archives(
        self, tenant_id: str, user_id: str, limit: int = 20
    ) -> List[dict]:
        ...

    async def fetch_cold_session(self, session_id: str) -> Optional[dict]:
        ...
