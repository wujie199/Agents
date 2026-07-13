"""SevenStepChunker — 实现 ChunkerPort，对接 IndexService。"""

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional

from core.ports.chunker import Chunk, ChunkerPort
from document.rag.config.chunk_pipeline import ChunkPipelineConfig, parse_chunk_pipeline_config
from document.rag.application.chunking.models import ScoredChunk
from document.rag.application.chunking.pipeline import SevenStepChunkPipeline

_logger = logging.getLogger("rag.chunking.seven_step_chunker")


def _to_chunk(
    scored: ScoredChunk,
    doc_id: str,
    idx: int,
    base_metadata: Dict[str, Any],
) -> Chunk:
    chunk_id = scored.metadata.get("chunk_id") or _generate_chunk_id(doc_id, idx, scored.content)
    meta = {
        **base_metadata,
        **scored.metadata,
        "strategy": "seven_step",
        "chunk_role": scored.chunk_role,
        "heading_path": scored.heading_path,
        "section_path": scored.heading_path,
        "unit_type": scored.unit_type,
        "density": scored.density,
        "quality": scored.quality,
        "score": scored.score,
        "dimension_scores": scored.dimension_scores,
        "entities": scored.entities,
    }
    if scored.parent_id:
        meta["parent_id"] = scored.parent_id
    for key in ("page", "block_type", "bbox", "source"):
        if key in scored.metadata:
            meta[key] = scored.metadata[key]
    if scored.inherited_entities:
        meta["inherited_entities"] = scored.inherited_entities
    if scored.contextualized_content:
        meta["_embed_content"] = scored.contextualized_content
    return Chunk(
        chunk_id=str(chunk_id),
        content=scored.content,
        doc_id=doc_id,
        chunk_index=idx,
        metadata=meta,
        char_count=len(scored.content),
    )


def _generate_chunk_id(doc_id: str, idx: int, content: str) -> str:
    h = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{doc_id}_chunk_{idx}_{h}"


class SevenStepChunker(ChunkerPort):
    """七步分块 Chunker，供 IndexService 调用。"""

    def __init__(
        self,
        pipeline_cfg: Optional[ChunkPipelineConfig] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        **_: Any,
    ):
        self._cfg = pipeline_cfg or ChunkPipelineConfig()
        self._embed_fn = embed_fn
        self._pipeline = SevenStepChunkPipeline(self._cfg, embed_fn=embed_fn)
        self._last_parents: List[ScoredChunk] = []
        self._last_stats: Dict[str, Any] = {}

    @classmethod
    def from_rag_config(cls, rag_cfg: Any, embed_fn: Optional[Callable] = None) -> "SevenStepChunker":
        raw = getattr(rag_cfg, "chunk_pipeline", None)
        if isinstance(raw, dict):
            pipe_cfg = parse_chunk_pipeline_config(raw)
        elif hasattr(rag_cfg, "chunk_pipeline"):
            pipe_cfg = rag_cfg.chunk_pipeline
        else:
            pipe_cfg = ChunkPipelineConfig()
        return cls(pipeline_cfg=pipe_cfg, embed_fn=embed_fn)

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        meta = dict(metadata or {})
        result = self._pipeline.run(text, doc_id, meta)
        self._last_parents = result.parent_chunks
        self._last_stats = result.stats
        base_meta = dict(meta)
        return [
            _to_chunk(sc, doc_id, idx, base_meta)
            for idx, sc in enumerate(result.retrieval_chunks)
        ]

    def chunk_batch(
        self,
        texts: List[str],
        doc_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[List[Chunk]]:
        return [self.chunk(text, doc_id, metadata) for text, doc_id in zip(texts, doc_ids)]

    def get_parent_chunks(self) -> List[ScoredChunk]:
        return list(self._last_parents)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        return dict(self._last_stats)
