#!/usr/bin/env python3
"""Memory evaluation CLI — session_search / L1 / HITL / L4 golden sets."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from document.memory.evaluation.run import run_memory_eval_job

_GOLDEN_DIR = REPO_ROOT / "data" / "memory_eval" / "golden"


def _resolve_dataset_path(ds: str) -> Path:
    dataset = Path(ds)
    if not dataset.is_absolute():
        dataset = REPO_ROOT / dataset
    if dataset.is_file():
        return dataset
    name = Path(ds).name
    for candidate in (name, f"{name}.jsonl" if not name.endswith(".jsonl") else name):
        golden = _GOLDEN_DIR / candidate
        if golden.is_file():
            return golden
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Memory golden evaluation")
    parser.add_argument(
        "--dataset",
        default="data/memory_eval/golden/enterprise.jsonl",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Run multiple golden files sequentially (overrides --dataset)",
    )
    parser.add_argument("--run-id", default="memory-eval")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use ModelRegistry for l1_extract; falls back to mock_extract if unavailable",
    )
    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="Force mock_extract for l1_extract rows",
    )
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    datasets = args.datasets or [args.dataset]
    overall_pass = True
    summaries = []
    for i, ds in enumerate(datasets):
        dataset = _resolve_dataset_path(ds)
        run_id = args.run_id if len(datasets) == 1 else f"{args.run_id}-{i+1}"
        data_dir = args.data_dir
        if not Path(data_dir).is_absolute():
            data_dir = str(REPO_ROOT / data_dir)
        result = asyncio.run(
            run_memory_eval_job(
                dataset_path=str(dataset),
                run_id=run_id,
                data_dir=data_dir,
                sample_limit=args.sample_limit or None,
                use_llm=args.use_llm,
                use_mock_only=args.mock_only or not args.use_llm,
                config_dir=args.config_dir,
            )
        )
        summaries.append({"dataset": str(dataset), **result})
        if result.get("pass_rate", 0) < 1.0:
            overall_pass = False
    print(json.dumps(summaries if len(summaries) > 1 else summaries[0], ensure_ascii=False, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
