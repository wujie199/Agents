# -*- coding: utf-8 -*-
"""Retrieve → generate → row assembly for RAGAS evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from core.composition.run_context import RunContext
from core.domain.evidence import EvidenceBundle

from app.agents.prompts.prompt_builder import build_chat_messages, format_evidence_bundle
from app.agents.roles.react_loop import _extract_llm_text
from document.rag.evaluation.dataset import EvalSample
from document.rag.evaluation.stack_factory import (
    EvalStackConfig,
    resolve_sample_rag_request,
)

EvalMode = Literal["full", "retrieval_only"]


@dataclass
class PipelineRow:
    id: str
    question: str
    ground_truth: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    reference_contexts: list[str] = field(default_factory=list)
    retrieval_empty: bool = False
    retrieval_degraded: bool = False
    degraded_reason: str | None = None
    error_code: str | None = None
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    tags: list[str] = field(default_factory=list)
    evidence_count: int = 0
    error: str | None = None
    metric_scores: dict[str, float] = field(default_factory=dict)

    def ragas_row(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "reference": self.ground_truth,
        }

    def to_detail_dict(self, *, preview_chars: int = 500) -> dict[str, Any]:
        def _preview(text: str) -> str:
            text = text or ""
            if len(text) <= preview_chars:
                return text
            return text[: preview_chars - 1] + "…"

        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "answer_preview": _preview(self.answer),
            "contexts_preview": [_preview(c) for c in self.contexts[:5]],
            "reference_contexts": self.reference_contexts,
            "retrieval_empty": self.retrieval_empty,
            "retrieval_degraded": self.retrieval_degraded,
            "degraded_reason": self.degraded_reason,
            "error_code": self.error_code,
            "retrieval_ms": round(self.retrieval_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "evidence_count": self.evidence_count,
            "tags": self.tags,
            "error": self.error,
            "metric_scores": dict(self.metric_scores),
        }


def _bundle_contexts(bundle: EvidenceBundle) -> list[str]:
    return [(ev.content or "").strip() for ev in (bundle.evidences or []) if (ev.content or "").strip()]


def _bundle_status(bundle: EvidenceBundle) -> tuple[bool, bool, str | None, str | None]:
    empty = bool(bundle.empty or not bundle.evidences)
    degraded = bundle.is_degraded()
    reason = bundle.degraded_reason.value if bundle.degraded_reason else None
    return empty, degraded, reason, bundle.error_code


async def retrieve_for_sample(
    ctx: RunContext,
    sample: EvalSample,
    *,
    profile: str,
    data_dir: str,
) -> tuple[EvidenceBundle, float]:
    rag_request = resolve_sample_rag_request(
        ctx, sample.tenant_id, profile=profile, data_dir=data_dir
    )
    t0 = time.perf_counter()
    if ctx.rag is None:
        from core.domain.evidence import DegradedReason

        bundle = EvidenceBundle.empty_bundle(
            DegradedReason.VECTOR_UNAVAILABLE, "rag_not_configured"
        )
    else:
        bundle = await ctx.rag.route_and_retrieve(
            sample.question,
            rag_request,
            plan=sample.retrieval_plan,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return bundle, elapsed_ms


async def generate_answer(
    ctx: RunContext,
    sample: EvalSample,
    bundle: EvidenceBundle,
    eval_cfg: EvalStackConfig,
) -> tuple[str, float]:
    gen = eval_cfg.generation
    evidence_text = format_evidence_bundle(
        bundle,
        max_chars=int(gen.get("max_evidence_chars", 6000)),
        max_items=int(gen.get("max_evidence_items", 8)),
        min_score=float(gen.get("rag_min_score", 0.0)),
        min_keep=int(gen.get("rag_min_keep", 3)),
        strict_grounding=bool(gen.get("strict_grounding", True)),
    )
    messages = build_chat_messages(
        memory_system=str(gen.get("memory_system") or "你是知识问答助手。"),
        user_message=sample.question,
        evidence_text=evidence_text,
    )
    role = str(gen.get("model_role") or "main_llm")
    llm = ctx.get_model(role)
    t0 = time.perf_counter()
    response = await llm.ainvoke(messages)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return _extract_llm_text(response), elapsed_ms


async def run_pipeline_sample(
    ctx: RunContext,
    sample: EvalSample,
    *,
    mode: EvalMode,
    eval_cfg: EvalStackConfig,
    profile: str,
    data_dir: str,
) -> PipelineRow:
    row = PipelineRow(
        id=sample.id,
        question=sample.question,
        ground_truth=sample.ground_truth,
        reference_contexts=list(sample.reference_contexts),
        tags=list(sample.tags),
    )
    try:
        bundle, retrieval_ms = await retrieve_for_sample(
            ctx, sample, profile=profile, data_dir=data_dir
        )
        row.retrieval_ms = retrieval_ms
        row.contexts = _bundle_contexts(bundle)
        row.evidence_count = len(row.contexts)
        empty, degraded, reason, error_code = _bundle_status(bundle)
        row.retrieval_empty = empty
        row.retrieval_degraded = degraded
        row.degraded_reason = reason
        row.error_code = error_code

        if mode == "full":
            answer, gen_ms = await generate_answer(ctx, sample, bundle, eval_cfg)
            row.answer = answer
            row.generation_ms = gen_ms
    except Exception as exc:
        row.error = str(exc)
    return row


async def run_pipeline(
    ctx: RunContext,
    samples: list[EvalSample],
    *,
    mode: EvalMode,
    eval_cfg: EvalStackConfig,
    profile: str,
    data_dir: str,
) -> list[PipelineRow]:
    rows: list[PipelineRow] = []
    for sample in samples:
        row = await run_pipeline_sample(
            ctx,
            sample,
            mode=mode,
            eval_cfg=eval_cfg,
            profile=profile,
            data_dir=data_dir,
        )
        rows.append(row)
    return rows
