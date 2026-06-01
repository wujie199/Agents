"""Backward compat: IndexPortAdapter delegates to IndexService."""

from typing import Any, Dict, List, Optional

from core.ports.chunker import ChunkStrategy
from core.ports.storage.cache import CachePort
from core.ports.storage.vector import VectorPort

from document.rag.config import RagPipelineConfig, load_rag_pipeline_config
from document.rag.application.indexing.service import IndexService


class IndexPortAdapter(IndexService):
    """Legacy name; prefer IndexService in new code."""

    def __init__(
        self,
        vector_port: VectorPort,
        embedding_model: Any,
        chunker: Optional[Any] = None,
        cache_port: Optional[CachePort] = None,
        chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 32,
        model_version: str = "v1",
        config: Optional[RagPipelineConfig] = None,
    ):
        cfg = config or load_rag_pipeline_config()
        if chunk_size != 500:
            cfg.chunk_size = chunk_size
        if chunk_overlap != 50:
            cfg.chunk_overlap = chunk_overlap
        cfg.model_version = model_version
        cfg.embedding_batch_size = batch_size
        super().__init__(
            vector_port=vector_port,
            embedding_model=embedding_model,
            config=cfg,
            cache_port=cache_port,
            chunk_strategy=chunk_strategy,
        )
