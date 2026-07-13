"""Step 3：向量 L2 校验与零向量过滤。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from document.rag.config.embedding import EmbeddingConfig

_log = logging.getLogger("rag.embedding.normalizer")


@dataclass
class NormalizeResult:
    vectors: List[List[float]] = field(default_factory=list)
    rejected_indices: List[int] = field(default_factory=list)
    reject_reasons: List[str] = field(default_factory=list)


def _l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vector))


def normalize_vector(
    vector: Sequence[float],
    cfg: EmbeddingConfig,
) -> Optional[List[float]]:
    """校验单位向量；必要时显式 L2 归一化。"""
    if not vector:
        return None
    vals = [float(x) for x in vector]
    norm = _l2_norm(vals)
    if norm < 1e-12:
        if cfg.reject_zero_vectors:
            return None
        return vals
    if cfg.verify_unit_norm:
        if abs(norm - 1.0) > float(cfg.unit_norm_tolerance):
            if cfg.force_l2_normalize:
                vals = [v / norm for v in vals]
            elif cfg.reject_zero_vectors:
                _log.debug("向量范数偏离 1.0: %.6f", norm)
    elif cfg.force_l2_normalize:
        vals = [v / norm for v in vals]
    return vals


def normalize_vectors(
    vectors: Sequence[Sequence[float]],
    cfg: EmbeddingConfig,
) -> NormalizeResult:
    out = NormalizeResult()
    for idx, vec in enumerate(vectors):
        normalized = normalize_vector(vec, cfg)
        if normalized is None:
            out.rejected_indices.append(idx)
            out.reject_reasons.append("zero_vector")
            continue
        out.vectors.append(normalized)
    return out
