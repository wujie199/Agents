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
    def execute(
        self,
        query: str,
        params: Optional[dict] = None
    ) -> Any:
        ...
    
    def execute_batch(
        self,
        queries: List[tuple[str, dict]]
    ) -> List[Any]:
        ...
    
    def insert(
        self,
        table: str,
        data: dict
    ) -> str:
        ...
    
    def update(
        self,
        table: str,
        data: dict,
        where: dict
    ) -> int:
        ...
    
    def delete(
        self,
        table: str,
        where: dict
    ) -> int:
        ...
    
    def select_one(
        self,
        table: str,
        columns: List[str],
        where: dict
    ) -> Optional[dict]:
        ...
    
    def select_many(
        self,
        table: str,
        columns: List[str],
        where: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[dict]:
        ...
    
    def begin_transaction(self) -> Any:
        ...
    
    def commit(self, transaction: Any) -> None:
        ...
    
    def rollback(self, transaction: Any) -> None:
        ...
