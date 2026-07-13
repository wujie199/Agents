"""Step 2：分批编码，OOM 时 batch 减半重试。"""

from __future__ import annotations

import logging
from typing import List, Sequence

from core.ports.rag.embedding import EmbeddingPort
from document.rag.config.embedding import EmbeddingConfig

_log = logging.getLogger("rag.embedding.batch_encode")

_OOM_MARKERS = ("out of memory", "cuda", "oom")


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _OOM_MARKERS)


async def _encode_once(model: EmbeddingPort, texts: Sequence[str]) -> List[List[float]]:
    if hasattr(model, "aembed"):
        return await model.aembed(list(texts))
    if hasattr(model, "embed"):
        return model.embed(list(texts))
    raise RuntimeError("Embedding model has no embed method")


async def encode_batches(
    model: EmbeddingPort,
    texts: Sequence[str],
    cfg: EmbeddingConfig,
) -> List[List[float]]:
    """按 batch_size 分片编码；OOM 时可选减半重试。"""
    if not texts:
        return []
    batch_size = max(1, int(cfg.batch_size))
    min_batch = max(1, int(cfg.batch_size_min))
    results: List[List[float]] = []
    idx = 0
    while idx < len(texts):
        current_bs = min(batch_size, len(texts) - idx)
        while True:
            batch = list(texts[idx : idx + current_bs])
            try:
                vectors = await _encode_once(model, batch)
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        f"编码数量不匹配: input={len(batch)} output={len(vectors)}"
                    )
                results.extend(vectors)
                idx += current_bs
                break
            except Exception as exc:
                if cfg.oom_halve_retry and _is_oom(exc) and current_bs > min_batch:
                    next_bs = max(min_batch, current_bs // 2)
                    _log.warning(
                        "Embedding OOM，batch %d → %d 重试",
                        current_bs,
                        next_bs,
                    )
                    current_bs = next_bs
                    continue
                raise
    return results

