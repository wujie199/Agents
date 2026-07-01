"""Backward compat: IndexPortAdapter delegates to IndexService."""

from dataclasses import replace
from typing import Any, Dict, List, Optional

from core.ports.chunker import ChunkStrategy
from core.ports.storage.cache import CachePort
from core.ports.storage.vector import VectorPort

from document.rag.config import RagPipelineConfig
from document.rag.application.indexing.service import IndexService


class IndexPortAdapter(IndexService):
    """Legacy name; prefer IndexService in new code."""

    def __init__(
        self,
        vector_port: VectorPort,
        embedding_model: Any,
        config: RagPipelineConfig,
        chunker: Optional[Any] = None,
        cache_port: Optional[CachePort] = None,
        chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 32,
        model_version: str = "v1",
    ):
        overrides: dict = {}
        if chunk_size != 500:
            overrides["chunk_size"] = chunk_size
        if chunk_overlap != 50:
            overrides["chunk_overlap"] = chunk_overlap
        overrides["model_version"] = model_version
        overrides["embedding_batch_size"] = batch_size
        cfg = replace(config, **overrides) if overrides else config
        super().__init__(
            vector_port=vector_port,
            embedding_model=embedding_model,
            config=cfg,
            cache_port=cache_port,
            chunk_strategy=chunk_strategy,
        )
