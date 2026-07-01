# -*- coding: utf-8 -*-
"""RAGAS metric profiles."""

from __future__ import annotations

from typing import Any, Literal

from document.rag.evaluation.stack_factory import EvalStackConfig

EvalMode = Literal["full", "retrieval_only"]

_EVAL_INSTALL_HINT = (
    "Install evaluation dependencies: pip install -e '.[eval]' "
    "(needs langchain-community>=0.3.31,<0.4 with langchain-core>=1.3)."
)


def check_ragas_import() -> str | None:
    """Return an error message if ragas cannot be imported, else None."""
    try:
        import ragas  # noqa: F401
        from ragas.metrics import Faithfulness  # noqa: F401
    except ImportError as exc:
        return f"{exc}. {_EVAL_INSTALL_HINT}"
    return None


METRIC_ALIASES = {
    "context_precision": "context_precision",
    "context_recall": "context_recall",
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "answer_correctness": "answer_correctness",
    "context_relevancy": "context_relevancy",
}


def metric_names_for_mode(mode: EvalMode, eval_cfg: EvalStackConfig) -> list[str]:
    key = "full" if mode == "full" else "retrieval_only"
    names = eval_cfg.metrics.get(key) or []
    return [METRIC_ALIASES.get(n, n) for n in names]


def build_ragas_metrics(
    mode: EvalMode,
    eval_cfg: EvalStackConfig,
    *,
    llm: Any,
    embeddings: Any | None = None,
) -> list[Any]:
    """Instantiate RAGAS metric objects for the given mode."""
    try:
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise ImportError(f"ragas import failed: {exc}. {_EVAL_INSTALL_HINT}") from exc

    names = set(metric_names_for_mode(mode, eval_cfg))
    metrics: list[Any] = []

    def _add(metric_cls: Any, *, needs_emb: bool = False) -> None:
        if needs_emb and embeddings is not None:
            metrics.append(metric_cls(llm=llm, embeddings=embeddings))
        elif needs_emb:
            metrics.append(metric_cls(llm=llm))
        else:
            metrics.append(metric_cls(llm=llm))

    if "context_precision" in names:
        _add(ContextPrecision)
    if "context_recall" in names:
        _add(ContextRecall)
    if "faithfulness" in names:
        _add(Faithfulness)
    if "answer_relevancy" in names:
        _add(AnswerRelevancy, needs_emb=True)
    if "answer_correctness" in names:
        _add(AnswerCorrectness, needs_emb=True)
    if "context_relevancy" in names:
        try:
            from ragas.metrics import ContextRelevancy

            _add(ContextRelevancy, needs_emb=True)
        except ImportError:
            pass

    return metrics


def run_ragas_evaluate(rows: list[dict[str, Any]], metrics: list[Any]) -> Any:
    """Run ragas.evaluate on assembled rows."""
    from datasets import Dataset
    from ragas import evaluate

    dataset = Dataset.from_list(rows)
    return evaluate(dataset, metrics=metrics)


def extract_metric_means(result: Any) -> dict[str, float | None]:
    """Extract mean scores from ragas EvaluationResult."""
    out: dict[str, float | None] = {}
    try:
        df = result.to_pandas()
        for col in df.columns:
            if col in ("question", "answer", "contexts", "ground_truth", "reference"):
                continue
            series = df[col]
            try:
                out[col] = float(series.mean())
            except (TypeError, ValueError):
                out[col] = None
    except Exception:
        scores = getattr(result, "scores", None) or {}
        for key, val in scores.items():
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                out[key] = None
    return out
