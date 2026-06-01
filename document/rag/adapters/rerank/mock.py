from typing import Any, Dict, List

from core.ports.rag.rerank import RerankPort


class MockRerankModel(RerankPort):
    """Dev/test reranker: token overlap with query."""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        q_tokens = set(query.lower().split())
        scored = []
        for idx, doc in enumerate(documents):
            d_tokens = set((doc or "").lower().split())
            overlap = len(q_tokens & d_tokens)
            scored.append((idx, float(overlap) + 0.01 * (len(documents) - idx)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"index": idx, "score": score}
            for idx, score in scored[:top_n]
        ]
