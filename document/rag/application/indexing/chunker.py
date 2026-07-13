"""Chunker 工厂 — 生产路径仅 seven_step。"""

from __future__ import annotations

import logging
from typing import Any

from core.ports.chunker import ChunkStrategy, ChunkerPort

_log = logging.getLogger("document.rag.indexing.chunker")

_SUPPORTED = frozenset({ChunkStrategy.SEVEN_STEP})


def create_chunker(
    strategy: ChunkStrategy = ChunkStrategy.SEVEN_STEP,
    **kwargs: Any,
) -> ChunkerPort:
    """构建分块器；仅支持 SEVEN_STEP，其它 strategy 会告警并回退。"""
    if strategy not in _SUPPORTED:
        _log.warning(
            "chunk_strategy=%s 已废弃，回退 seven_step", strategy.value
        )
        strategy = ChunkStrategy.SEVEN_STEP

    from document.rag.application.chunking.chunker import SevenStepChunker

    return SevenStepChunker(**kwargs)


def parse_chunk_strategy(name: str) -> ChunkStrategy:
    key = (name or "seven_step").lower().strip()
    if key == "seven_step":
        return ChunkStrategy.SEVEN_STEP
    _log.warning("chunk_strategy=%r 已废弃，使用 seven_step", key)
    return ChunkStrategy.SEVEN_STEP
