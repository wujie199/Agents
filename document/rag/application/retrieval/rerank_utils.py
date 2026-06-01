import logging
from typing import Any, List, Optional

from core.domain.evidence import Evidence


async def apply_rerank(
    rerank_model: Any,
    query: str,
    evidences: List[Evidence],
    top_n: int,
    logger: Optional[logging.Logger] = None,
) -> List[Evidence]:
    """Module docstring."""
    log = logger or logging.getLogger("rag.retrieval.rerank")
    if not evidences or rerank_model is None or top_n <= 0:
        return evidences[:top_n] if top_n else evidences

    try:
        documents = [e.content for e in evidences]
        if not hasattr(rerank_model, "rerank"):
            return evidences[:top_n]

        raw = rerank_model.rerank(query, documents, top_n=top_n)
        if not raw:
            return evidences[:top_n]

        reranked: List[Evidence] = []
        for item in raw:
            if isinstance(item, dict):
                idx = int(item.get("index", 0))
                score = float(item.get("score", 0.0))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                idx, score = int(item[0]), float(item[1])
            else:
                continue
            if idx >= len(evidences):
                continue
            e = evidences[idx]
            reranked.append(
                Evidence(
                    id=e.id,
                    content=e.content,
                    score=score,
                    source_type=e.source_type,
                    citation=e.citation,
                    metadata={**e.metadata, "rerank_score": score},
                )
            )
        return reranked[:top_n] if reranked else evidences[:top_n]
    except Exception as exc:
        log.warning("Rerank failed: %s", exc)
        return evidences[:top_n]
