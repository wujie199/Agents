# -*- coding: utf-8 -*-
"""Golden dataset load/validate for RAG evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class EvalSample:
    id: str
    question: str
    ground_truth: str
    reference_contexts: list[str] = field(default_factory=list)
    tenant_id: str | None = None
    retrieval_plan: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "reference_contexts": list(self.reference_contexts),
            "tenant_id": self.tenant_id,
            "retrieval_plan": self.retrieval_plan,
            "tags": list(self.tags),
        }


class DatasetValidationError(ValueError):
    pass


def _validate_row(raw: dict[str, Any], line_no: int) -> EvalSample:
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"line {line_no}: row must be a JSON object")
    sample_id = str(raw.get("id") or "").strip()
    question = str(raw.get("question") or "").strip()
    ground_truth = str(raw.get("ground_truth") or "").strip()
    if not sample_id:
        raise DatasetValidationError(f"line {line_no}: missing id")
    if not question:
        raise DatasetValidationError(f"line {line_no}: missing question")
    if not ground_truth:
        raise DatasetValidationError(f"line {line_no}: missing ground_truth")

    ref_ctx = raw.get("reference_contexts") or []
    if ref_ctx is not None and not isinstance(ref_ctx, list):
        raise DatasetValidationError(
            f"line {line_no}: reference_contexts must be a list"
        )
    reference_contexts = [str(x).strip() for x in ref_ctx if str(x).strip()]

    tags_raw = raw.get("tags") or []
    if tags_raw is not None and not isinstance(tags_raw, list):
        raise DatasetValidationError(f"line {line_no}: tags must be a list")
    tags = [str(x).strip() for x in tags_raw if str(x).strip()]

    plan = raw.get("retrieval_plan")
    if plan is not None and not isinstance(plan, dict):
        raise DatasetValidationError(f"line {line_no}: retrieval_plan must be a dict")

    tenant_id = raw.get("tenant_id")
    if tenant_id is not None:
        tenant_id = str(tenant_id).strip() or None

    return EvalSample(
        id=sample_id,
        question=question,
        ground_truth=ground_truth,
        reference_contexts=reference_contexts,
        tenant_id=tenant_id,
        retrieval_plan=plan,
        tags=tags,
    )


def load_eval_dataset(path: str | Path) -> list[EvalSample]:
    """Load JSONL golden dataset."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"dataset not found: {p}")

    samples: list[EvalSample] = []
    seen_ids: set[str] = set()
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"line {line_no}: invalid JSON: {exc}"
                ) from exc
            sample = _validate_row(raw, line_no)
            if sample.id in seen_ids:
                raise DatasetValidationError(
                    f"line {line_no}: duplicate id {sample.id!r}"
                )
            seen_ids.add(sample.id)
            samples.append(sample)
    if not samples:
        raise DatasetValidationError(f"dataset empty: {p}")
    return samples


def iter_limited(samples: list[EvalSample], limit: int | None) -> Iterator[EvalSample]:
    if limit is None or limit <= 0:
        yield from samples
        return
    yield from samples[:limit]
