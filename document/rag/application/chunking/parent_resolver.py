"""检索时按 parent_id 扩展父 Chunk 上下文。"""

from typing import Dict, List, Optional, Set

from core.domain.evidence import Evidence
from document.rag.application.chunking.parent_store import ParentChunkStore


def expand_evidences_with_parent_context(
    evidences: List[Evidence],
    collection: str,
    parent_store: Optional[ParentChunkStore] = None,
    max_parents: int = 5,
) -> List[Evidence]:
    """子 chunk 命中后拉取父 chunk 全文，去重后用于 LLM 生成。"""
    if not evidences or parent_store is None:
        return evidences

    seen_parents: Set[str] = set()
    expanded: List[Evidence] = []

    for ev in evidences:
        meta = dict(ev.metadata or {})
        parent_id = meta.get("parent_id")
        doc_id = meta.get("doc_id")
        if not parent_id or not doc_id:
            expanded.append(ev)
            continue
        if parent_id in seen_parents:
            continue
        if len(seen_parents) >= max_parents:
            expanded.append(ev)
            continue

        parent_content = parent_store.get_by_id(collection, doc_id, parent_id)
        if not parent_content:
            expanded.append(ev)
            continue

        seen_parents.add(parent_id)
        expanded.append(
            Evidence(
                id=str(parent_id),
                content=parent_content,
                source_type=ev.source_type,
                score=ev.score,
                citation=ev.citation,
                metadata={**meta, "chunk_role": "parent", "expanded_from_child": ev.id},
            )
        )

    return expanded or evidences
