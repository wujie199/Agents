from typing import Any, Dict, List, Protocol


class RerankPort(Protocol):
    """RAG 内部：证据重排。实现可来自 ModelPort.get_reranker() 或 Mock。"""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        ...
