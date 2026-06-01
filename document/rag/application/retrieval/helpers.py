import asyncio
from typing import Any, Dict, List, Optional

from core.domain.evidence import Evidence, SourceType
from core.ports.storage.vector import VectorPort


def source_type_from_backend(backend: str) -> SourceType:
    mapping = {
        "vector": SourceType.VECTOR,
        "sql": SourceType.SQL,
        "graph": SourceType.GRAPH,
        "redis_cache": SourceType.CACHE,
        "cache": SourceType.CACHE,
    }
    return mapping.get(backend, SourceType.VECTOR)


def check_acl(metadata: Dict, acl: Any) -> bool:
    if acl is None:
        return True
    doc_id = metadata.get("doc_id")
    if not doc_id or not hasattr(acl, "can_access_doc"):
        return True
    doc_ids = getattr(acl, "doc_ids", None)
    if doc_ids is not None and len(doc_ids) == 0:
        return True
    return acl.can_access_doc(doc_id)


def search_results_to_evidences(
    results: List[Any],
    source_type: SourceType = SourceType.VECTOR,
) -> List[Evidence]:
    evidences = []
    for r in results:
        if isinstance(r, dict):
            meta = dict(r.get("metadata") or {})
            evidences.append(
                Evidence(
                    id=str(r.get("id") or meta.get("chunk_id") or ""),
                    content=str(r.get("content") or ""),
                    source_type=source_type,
                    score=float(r.get("score", 0.0)),
                    citation=meta.get("source_path"),
                    metadata=meta,
                )
            )
            continue
        meta = getattr(r, "metadata", None) or {}
        evidences.append(
            Evidence(
                id=getattr(r, "id", meta.get("chunk_id", "")),
                content=getattr(r, "content", None) or "",
                source_type=source_type,
                score=float(getattr(r, "score", 0.0)),
                citation=meta.get("source_path"),
                metadata=meta,
            )
        )
    return evidences


async def vector_similarity_search(
    vector_port: VectorPort,
    collection: str,
    query_vector: List[float],
    top_k: int,
    tenant_id: str,
    acl: Any,
) -> List[Evidence]:
    results = await asyncio.to_thread(
        vector_port.similarity_search,
        collection,
        query_vector,
        top_k,
        {"tenant_id": tenant_id},
    )
    evidences = []
    for r in results:
        meta = r.metadata or {}
        if not check_acl(meta, acl):
            continue
        evidences.append(
            Evidence(
                id=r.id,
                content=r.content or "",
                source_type=SourceType.VECTOR,
                score=r.score,
                citation=meta.get("source_path"),
                metadata=meta,
            )
        )
    return evidences


async def cache_get(cache_port: Any, key: str) -> Any:
    value = cache_port.get(key)
    if asyncio.iscoroutine(value):
        return await value
    return value


async def cache_set(cache_port: Any, key: str, value: Any, ttl: Optional[int] = None) -> None:
    result = cache_port.set(key, value, ttl)
    if asyncio.iscoroutine(result):
        await result
