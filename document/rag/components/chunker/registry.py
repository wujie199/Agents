"""Chunker 独立 registry — 按策略构建 ChunkerPort。"""

from core.ports.rag.chunker import ChunkerPort
from document.rag.config.pipeline import RagPipelineConfig


def build_chunker(cfg: RagPipelineConfig) -> ChunkerPort:
    """按 config.chunk_strategy 构建 chunker（当前仅 faq/recursive 策略）。"""
    strategy = (cfg.chunk_strategy or "faq").lower()
    # 当前 chunker 在 indexing/service 中内联使用 split_text_into_chunks，
    # 后续可在此处切换实现。
    from document.rag.shared.text_chunker import split_text_into_chunks
    return split_text_into_chunks
