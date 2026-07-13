"""Step7：去重与重叠控制。"""

from hashlib import md5
from typing import Callable, List, Optional, Set

from document.rag.config.chunk_pipeline import ChunkPipelineConfig
from document.rag.application.chunking.models import ScoredChunk
from document.rag.shared.dedupe import _cosine_similarity


def _exact_hash(text: str) -> str:
    return md5(text.encode("utf-8")).hexdigest()


def run_step7_dedupe(
    chunks: List[ScoredChunk],
    cfg: ChunkPipelineConfig,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    has_parent_child: bool = False,
) -> List[ScoredChunk]:
    """精确去重 + 语义去重 + 边界重叠控制 + 父子关系清理。"""
    if not chunks:
        return chunks

    # L1 精确去重
    seen_hash: Set[str] = set()
    deduped: List[ScoredChunk] = []
    for ch in chunks:
        if not cfg.enable_exact_dedupe:
            deduped.append(ch)
            continue
        h = _exact_hash(ch.content)
        if h in seen_hash:
            continue
        seen_hash.add(h)
        deduped.append(ch)

    # L2 语义去重
    if cfg.enable_semantic_dedupe and len(deduped) > 1:
        if embed_fn is not None:
            texts = [c.content for c in deduped]
            try:
                vectors = embed_fn(texts)
                kept: List[ScoredChunk] = []
                kept_vecs: List[List[float]] = []
                for ch, vec in zip(deduped, vectors):
                    duplicate = False
                    for existing_vec, existing_ch in zip(kept_vecs, kept):
                        if _cosine_similarity(vec, existing_vec) >= cfg.semantic_dedup_threshold:
                            if ch.score > existing_ch.score:
                                existing_ch.content = ch.content
                                existing_ch.score = ch.score
                            duplicate = True
                            break
                    if not duplicate:
                        kept.append(ch)
                        kept_vecs.append(vec)
                deduped = kept
            except Exception:
                pass

    # L3 边界重叠（无父子层级时）
    if not has_parent_child and cfg.overlap_ratio > 0 and len(deduped) > 1:
        overlap_ratio = cfg.overlap_ratio
        for i in range(len(deduped) - 1):
            left = deduped[i]
            right = deduped[i + 1]
            overlap_size = int(min(left.size, right.size) * overlap_ratio)
            if overlap_size <= 0:
                continue
            tail = left.content[-overlap_size:]
            head = right.content[:overlap_size]
            left.metadata["overlap_tail"] = tail
            right.metadata["overlap_head"] = head

    # L4 清理无效 parent 引用
    valid_ids = {c.metadata.get("chunk_id") for c in deduped if c.metadata.get("chunk_id")}
    for ch in deduped:
        pid = ch.parent_id
        if pid and pid not in valid_ids:
            ch.parent_id = None
            ch.metadata.pop("parent_id", None)

    return deduped
