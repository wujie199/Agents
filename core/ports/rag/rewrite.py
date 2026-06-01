from typing import List, Protocol


class QueryRewritePort(Protocol):
    """RAG 内部：检索前查询改写（HyDE / Multi-Query）。不暴露给 L6/L7 业务。"""

    async def rewrite(self, query: str) -> List[str]:
        ...

    def is_enabled(self) -> bool:
        ...
