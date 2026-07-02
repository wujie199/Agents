# -*- coding: utf-8 -*-
"""Traditional IR metrics for RAG retrieval evaluation (Hit@k, MRR, NDCG, Recall@k)."""

from __future__ import annotations

import math
from typing import Any

from document.rag.evaluation.pipeline import PipelineRow
from document.rag.evaluation.text_norm import normalize_match_text, strip_leading_faq_number


def contexts_match(
    reference: str,
    retrieved: str,
    *,
    min_overlap_chars: int = 20,
) -> bool:
    """Heuristic match: normalized overlap (robust to punctuation / cleaning)."""
    a = normalize_match_text(reference)
    b = normalize_match_text(retrieved)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True

    overlap = min(min_overlap_chars, len(a), len(b))
    if overlap >= 8:
        if a[:overlap] in b or b[:overlap] in a:
            return True

    # 去掉题号后再比（reference 常带标点，index chunk 已清洗）
    a_body = strip_leading_faq_number(reference)
    b_body = strip_leading_faq_number(retrieved)
    if a_body and b_body:
        if a_body == b_body or a_body in b_body or b_body in a_body:
            return True
        overlap = min(min_overlap_chars, len(a_body), len(b_body))
        if overlap >= 8 and (a_body[:overlap] in b_body or b_body[:overlap] in a_body):
            return True

    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 16 and len(shorter) / len(longer) >= 0.55:
        probe = shorter[: min(32, len(shorter))]
        if probe in longer:
            return True

    return False


def _relevance_vector(
    references: list[str],
    retrieved: list[str],
    *,
    min_overlap_chars: int,
) -> list[int]:
    if not references:
        return [0] * len(retrieved)
    return [
        1
        if any(
            contexts_match(ref, ctx, min_overlap_chars=min_overlap_chars)
            for ref in references
        )
        else 0
        for ctx in retrieved
    ]


def hit_at_k(relevances: list[int], k: int) -> float:
    return 1.0 if any(relevances[:k]) else 0.0


def mrr_at_k(relevances: list[int], k: int) -> float:
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(relevances: list[int], k: int) -> float:
    rels = relevances[:k]
    if not rels or not any(rels):
        return 0.0

    def _dcg(values: list[int]) -> float:
        return sum(v / math.log2(i + 2) for i, v in enumerate(values))

    dcg = _dcg(rels)
    ideal = _dcg(sorted(rels, reverse=True))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(
    references: list[str],
    retrieved: list[str],
    k: int,
    *,
    min_overlap_chars: int,
) -> float:
    if not references:
        return 0.0
    matched = 0
    for ref in references:
        if any(
            contexts_match(ref, ctx, min_overlap_chars=min_overlap_chars)
            for ctx in retrieved[:k]
        ):
            matched += 1
    return matched / len(references)


def compute_sample_ir_scores(
    *,
    reference_contexts: list[str],
    retrieved_contexts: list[str],
    ks: list[int],
    min_overlap_chars: int = 20,
) -> dict[str, float]:
    refs = [r for r in reference_contexts if (r or "").strip()]
    ctxs = [c for c in retrieved_contexts if (c or "").strip()]
    rels = _relevance_vector(refs, ctxs, min_overlap_chars=min_overlap_chars)

    scores: dict[str, float] = {}
    for k in sorted(set(ks)):
        scores[f"hit@{k}"] = hit_at_k(rels, k)
        scores[f"mrr@{k}"] = mrr_at_k(rels, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(rels, k)
        scores[f"recall@{k}"] = recall_at_k(
            refs, ctxs, k, min_overlap_chars=min_overlap_chars
        )
    return scores


def default_ir_ks(eval_ir: dict[str, Any] | None) -> list[int]:
    raw = (eval_ir or {}).get("ks")
    if isinstance(raw, list) and raw:
        return sorted({int(x) for x in raw if int(x) > 0})
    return [1, 3, 5]


def aggregate_ir_metrics(
    rows: list[PipelineRow],
    *,
    ks: list[int] | None = None,
    min_overlap_chars: int = 20,
) -> dict[str, float]:
    ks = ks or [1, 3, 5]
    evaluable = [r for r in rows if not r.error]
    if not evaluable:
        return {}

    sums: dict[str, float] = {}
    for row in evaluable:
        sample_scores = compute_sample_ir_scores(
            reference_contexts=row.reference_contexts,
            retrieved_contexts=row.contexts,
            ks=ks,
            min_overlap_chars=min_overlap_chars,
        )
        row.ir_scores = sample_scores
        for name, val in sample_scores.items():
            sums[name] = sums.get(name, 0.0) + val

    n = len(evaluable)
    return {name: round(total / n, 4) for name, total in sorted(sums.items())}


def score_rows_ir_metrics(
    rows: list[PipelineRow],
    eval_ir: dict[str, Any] | None,
) -> dict[str, float]:
    cfg = eval_ir or {}
    if cfg.get("enabled") is False:
        return {}
    ks = default_ir_ks(cfg)
    min_overlap = int(cfg.get("min_overlap_chars") or 20)
    return aggregate_ir_metrics(rows, ks=ks, min_overlap_chars=min_overlap)
