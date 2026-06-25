"""会话搜索端口 — 全文检索 + 摘要。"""

from typing import Protocol

from core.domain.context import RequestContext
from core.ports.memory.dtos import SessionSearchResult


class SessionSearchPort(Protocol):
    """会话搜索契约：全文检索 + 结构化结果"""

    async def session_search(
        self,
        query: str,
        context: RequestContext,
        limit: int = 5,
        scope: str = "session",
    ) -> str:
        ...

    async def session_search_detail(
        self,
        query: str,
        context: RequestContext,
        limit: int = 5,
        scope: str = "session",
    ) -> SessionSearchResult:
        ...
