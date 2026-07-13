"""Chunker 独立 registry — 按策略构建 ChunkerPort。"""

from core.ports.rag.chunker import ChunkerPort
from document.rag.config.pipeline import RagPipelineConfig
from document.rag.application.indexing.chunker import create_chunker, parse_chunk_strategy


def build_chunker(cfg: RagPipelineConfig) -> ChunkerPort:
    """按 config.chunk_strategy 构建 chunker。"""
    strategy = parse_chunk_strategy(cfg.chunk_strategy)
    kwargs = {
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "pipeline_cfg": cfg.chunk_pipeline,
    }
    return create_chunker(strategy, **kwargs)
