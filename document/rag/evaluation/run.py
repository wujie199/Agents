# -*- coding: utf-8 -*-
"""Main RAG evaluation entry point."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from document.rag.evaluation.dataset import EvalSample, iter_limited, load_eval_dataset
from document.rag.evaluation.llm_bridge import build_ragas_models
from document.rag.evaluation.ir_metrics import score_rows_ir_metrics
from document.rag.evaluation.metrics import (
    build_ragas_metrics,
    extract_metric_means,
    run_ragas_evaluate,
)
from document.rag.evaluation.pipeline import PipelineRow, run_pipeline
from document.rag.evaluation.report import build_failures, build_summary, write_run_reports
from document.rag.evaluation.stack_factory import (
    build_eval_request,
    build_eval_run_context,
    load_eval_config,
)

EvalMode = Literal["full", "retrieval_only"]
EvalProfile = Literal["dev", "production"]

logger = logging.getLogger(__name__)


async def run_rag_eval(
    *,
    profile: EvalProfile = "dev",
    dataset_path: str = "data/rag_eval/golden/test_docs_merged.jsonl",
    mode: EvalMode = "full",
    run_id: str = "eval-run",
    sample_limit: int | None = None,
    config_dir: str = "config",
    data_dir: str = "data",
    skip_ragas: bool = False,
) -> dict[str, Any]:
    """Run end-to-end RAG evaluation and write reports."""
    eval_cfg = load_eval_config(config_dir)
    samples = load_eval_dataset(dataset_path)
    limited = list(iter_limited(samples, sample_limit))

    request = build_eval_request(
        sample_tenant_id=limited[0].tenant_id if limited else None,
        default_tenant="default",
        user_id="rag_eval",
        session_id=f"rag-eval-{run_id}",
        trace_id=f"rag-eval-{run_id}",
    )
    ctx = build_eval_run_context(
        request,
        profile=profile,
        config_dir=config_dir,
        data_dir=data_dir,
        disable_cache=True,
    )

    rows = await run_pipeline(
        ctx,
        limited,
        mode=mode,
        eval_cfg=eval_cfg,
        profile=profile,
        data_dir=data_dir,
    )

    metric_means: dict[str, float | None] = {}
    ir_metric_means: dict[str, float] = score_rows_ir_metrics(rows, eval_cfg.ir)
    ragas_error: str | None = None
    evaluable = [r for r in rows if not r.error]

    if not skip_ragas and evaluable:
        try:
            bundle = build_ragas_models(ctx.models, eval_cfg)
            metrics = build_ragas_metrics(
                mode,
                eval_cfg,
                llm=bundle.llm,
                embeddings=bundle.embeddings,
            )
            if metrics:
                ragas_rows = [r.ragas_row() for r in evaluable]
                result = run_ragas_evaluate(ragas_rows, metrics)
                metric_means = extract_metric_means(result)
                per_sample = {}
                try:
                    df = result.to_pandas()
                    for idx, row in enumerate(evaluable):
                        if idx >= len(df):
                            break
                        scores = {
                            col: float(df.iloc[idx][col])
                            for col in df.columns
                            if col
                            not in (
                                "question",
                                "answer",
                                "contexts",
                                "ground_truth",
                                "reference",
                            )
                            and df.iloc[idx][col] is not None
                        }
                        per_sample[row.id] = scores
                except Exception:
                    per_sample = {}
                for row in rows:
                    if row.id in per_sample:
                        row.metric_scores = per_sample[row.id]
        except ImportError as exc:
            ragas_error = str(exc)
            logger.warning("RAGAS not available: %s", exc)
        except Exception as exc:
            ragas_error = str(exc)
            logger.exception("RAGAS evaluation failed")

    preview_chars = int(eval_cfg.output.get("preview_chars", 500))
    results_root = Path(eval_cfg.output.get("results_dir") or "data/rag_eval/results")
    output_dir = results_root / run_id

    detail_rows = []
    for row in rows:
        detail = row.to_detail_dict(preview_chars=preview_chars)
        if row.metric_scores:
            detail["metric_scores"] = row.metric_scores
        if row.ir_scores:
            detail["ir_scores"] = row.ir_scores
        detail_rows.append(detail)

    summary = build_summary(
        run_id=run_id,
        mode=mode,
        profile=profile,
        dataset_path=dataset_path,
        sample_count=len(limited),
        rows=rows,
        metric_means=metric_means,
        ir_metric_means=ir_metric_means,
        config_meta={
            "config_dir": config_dir,
            "data_dir": data_dir,
            "generation_model_role": eval_cfg.generation.get("model_role", "main_llm"),
            "judge_model_role": eval_cfg.judge.get("model_role", "eval_judge_llm"),
            "ragas_error": ragas_error,
            "cache_disabled": True,
        },
    )

    failure_rows = build_failures(rows, preview_chars=preview_chars)
    report_paths = write_run_reports(
        output_dir,
        summary=summary,
        detail_rows=detail_rows,
        failure_rows=failure_rows,
    )

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "summary": summary,
        "report_paths": report_paths,
        "rows": rows,
        "ragas_error": ragas_error,
    }
