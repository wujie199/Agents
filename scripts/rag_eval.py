#!/usr/bin/env python3
"""RAGAS evaluation CLI — offline RAG pipeline quality assessment.

用法:
  python scripts/rag_eval.py --profile dev --dataset data/rag_eval/golden/test_docs_merged.jsonl --mode full --run-id test-run
  python scripts/rag_eval.py --profile dev --mode retrieval_only --sample-limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from document.rag.evaluation.metrics import check_ragas_import
from document.rag.evaluation.run import run_rag_eval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS RAG evaluation (offline job)")
    parser.add_argument(
        "--profile",
        choices=("dev", "production"),
        default="dev",
        help="RunContext profile (dev / production)",
    )
    parser.add_argument(
        "--dataset",
        default="data/rag_eval/golden/test_docs_merged.jsonl",
        help="Golden dataset JSONL path (default: merged test_docs txt)",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "retrieval_only"),
        default="full",
        help="full=retrieve+generate+metrics; retrieval_only=context metrics",
    )
    parser.add_argument(
        "--run-id",
        default="eval-run",
        help="Run identifier; reports written to data/rag_eval/results/{run_id}/",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Limit number of samples (0 = all)",
    )
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Config directory",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Run pipeline only, skip RAGAS metric scoring",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.skip_ragas:
        ragas_err = check_ragas_import()
        if ragas_err:
            print(f"Warning: RAGAS unavailable — {ragas_err}", file=sys.stderr)
    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = REPO_ROOT / dataset
    config_dir = args.config_dir
    if not Path(config_dir).is_absolute():
        config_dir = str(REPO_ROOT / config_dir)
    data_dir = args.data_dir
    if not Path(data_dir).is_absolute():
        data_dir = str(REPO_ROOT / data_dir)

    result = asyncio.run(
        run_rag_eval(
            profile=args.profile,
            dataset_path=str(dataset),
            mode=args.mode,
            run_id=args.run_id,
            sample_limit=args.sample_limit or None,
            config_dir=config_dir,
            data_dir=data_dir,
            skip_ragas=args.skip_ragas,
        )
    )

    summary = result["summary"]
    print(f"\nRAG evaluation complete: run_id={result['run_id']}")
    print(f"  output: {result['output_dir']}")
    print(f"  samples: {summary.get('sample_count')}")
    print(f"  failures: {json.dumps(summary.get('failures', {}), ensure_ascii=False)}")
    ir_metrics = summary.get("ir_metrics") or {}
    if ir_metrics:
        print("  ir_metrics:")
        for name, score in sorted(ir_metrics.items()):
            print(f"    {name}: {score:.4f}")
    metrics = summary.get("metrics") or {}
    if metrics:
        print("  ragas_metrics:")
        for name, score in sorted(metrics.items()):
            val = "—" if score is None else f"{score:.4f}"
            print(f"    {name}: {val}")
    elif result.get("ragas_error"):
        print(f"  ragas: skipped/failed — {result['ragas_error']}")
    elif not ir_metrics:
        print("  ragas: no metrics (use --skip-ragas=false with ragas installed)")
    print(f"  report: {result['report_paths'].get('report_html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
