from typing import Protocol, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SessionArchive:
    session_id: str
    user_id: str
    tenant_id: str
    channel: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: str = "active"


@dataclass
class MessageRecord:
    message_id: str
    session_id: str
    role: str
    content: str
    ts: datetime
    token_count: int = 0
    redacted: bool = False
    metadata: dict = field(default_factory=dict)


class RelationalPort(Protocol):
    async def execute(
        self,
        query: str,
        params: Optional[dict] = None,
    ) -> Any:
        ...

    async def execute_batch(
        self,
        queries: List[tuple[str, dict]],
    ) -> List[Any]:
        ...

    async def insert(
        self,
        table: str,
        data: dict,
    ) -> str:
        ...

    async def update(
        self,
        table: str,
        data: dict,
        where: dict,
    ) -> int:
        ...

    async def delete(
        self,
        table: str,
        where: dict,
    ) -> int:
        ...

    async def select_one(
        self,
        table: str,
        columns: List[str],
        where: dict,
    ) -> Optional[dict]:
        ...

    async def select_many(
        self,
        table: str,
        columns: List[str],
        where: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        ...

    async def upsert_session(self, data: dict) -> None:
        ...

    async def end_session(self, session_id: str, status: str = "closed") -> None:
        ...

    async def list_sessions(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> List[dict]:
        ...

    async def insert_message(self, data: dict) -> str:
        ...

    async def insert_tool_call(self, data: dict) -> str:
        ...

    async def search_messages(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        ...

    async def search_tool_calls(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        ...

    async def purge_expired_sessions(self, retention_days: int = 90) -> int:
        ...

    async def list_expired_session_ids(self, retention_days: int = 90) -> List[str]:
        ...

    async def anonymize_user_data(self, tenant_id: str, user_id: str) -> int:
        ...

    async def list_messages_for_reindex(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[dict]:
        ...

    async def delete_online_session(self, session_id: str) -> None:
        ...

    async def insert_cold_archive_index(self, data: dict) -> str:
        ...

    async def get_cold_archive(self, session_id: str) -> Optional[dict]:
        ...

    async def list_cold_archives(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> List[dict]:
        ...

    async def delete_cold_archive_index(self, session_id: str) -> int:
        ...

    async def begin_transaction(self) -> Any:
        ...

    async def commit(self, transaction: Any) -> None:
        ...

    async def rollback(self, transaction: Any) -> None:
        ...
