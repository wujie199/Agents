# -*- coding: utf-8 -*-
"""两段式检索：首轮结果质量判定。"""

from __future__ import annotations

from typing import List, Sequence

from document.rag.application.retrieval.query_intent import (
    detect_maintenance_intent,
    evidence_routing_score,
    is_maintenance_source,
)
from document.rag.config.rewrite import TwoStageConfig


def _top_score(evidence) -> float:
    return evidence_routing_score(evidence)


def retrieval_satisfactory(
    query: str,
    profile: str,
    evidences: Sequence,
    cfg: TwoStageConfig,
) -> bool:
    """首轮检索是否足够好，无需 L2 LLM 扩写。"""
    if not evidences:
        return False
    top = evidences[0]
    score = _top_score(top)
    if score < cfg.top1_min_rerank:
        return False
    if (
        profile == "maintenance"
        and cfg.require_maintenance_source
        and detect_maintenance_intent(query)
    ):
        meta = getattr(top, "metadata", None) or {}
        if not is_maintenance_source(meta):
            return False
    return True
