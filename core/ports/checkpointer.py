from typing import Protocol, Optional, List, Any


class CheckpointerPort(Protocol):
    """图状态快照（短生命周期），与 L2 Session Archive 职责分离。"""

    async def save(
        self,
        thread_id: str,
        tenant_id: str,
        state: dict,
        *,
        session_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        checkpoint_ns: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        ...

    async def load(
        self,
        thread_id: str,
        tenant_id: str,
        *,
        checkpoint_ns: str = "",
    ) -> Optional[dict]:
        ...

    async def list_threads(
        self, tenant_id: str, *, limit: int = 20
    ) -> List[dict]:
        ...
