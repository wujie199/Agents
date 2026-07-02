#!/usr/bin/env python3
"""生成 RAG 评测黄金集（PDF / TXT / 合并 test_docs）。

用法:
  # 合并 data/test_docs 下全部 txt（与当前 FaqChunker 建库块对齐）
  python scripts/build_rag_eval_dataset.py --merge-txt

  # 单个 txt
  python scripts/build_rag_eval_dataset.py --txt data/test_docs/扫地机器人100问2.txt

  # PDF（需存在 扫地机器人100问.pdf，输出需显式指定 --output 路径）
  python scripts/build_rag_eval_dataset.py --pdf data/test_docs/扫地机器人100问.pdf \\
      --output /tmp/vacuum_from_pdf.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from document.rag.evaluation.dataset_builder import (
    build_merged_from_txt_dir,
    build_pdf_dataset,
    build_samples_from_txt,
    rows_to_dicts,
    write_jsonl,
)

DEFAULT_TXT_DIR = REPO_ROOT / "data/test_docs"
DEFAULT_GOLDEN_OUTPUT = REPO_ROOT / "data/rag_eval/golden/test_docs_merged.jsonl"
DEFAULT_PDF = REPO_ROOT / "data/test_docs/扫地机器人100问.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build RAG eval golden JSONL from PDF/TXT (index-aligned)"
    )
    parser.add_argument(
        "--merge-txt",
        action="store_true",
        help="Merge all txt under --txt-dir into one golden set",
    )
    parser.add_argument(
        "--txt-dir",
        type=Path,
        default=DEFAULT_TXT_DIR,
        help="Directory of source txt files",
    )
    parser.add_argument(
        "--glob",
        default="*.txt",
        help="Glob under --txt-dir when --merge-txt",
    )
    parser.add_argument(
        "--txt",
        type=Path,
        default=None,
        help="Single txt file to convert",
    )
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL (default: test_docs_merged.jsonl for txt merge)",
    )
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--count", type=int, default=100, help="PDF row limit")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--profile", default="faq")
    parser.add_argument(
        "--no-align-index",
        action="store_true",
        help="Skip ingest+clean+chunk alignment (FAQ md fallback only)",
    )
    args = parser.parse_args()

    align_index = not args.no_align_index

    if args.merge_txt:
        txt_dir = args.txt_dir if args.txt_dir.is_absolute() else REPO_ROOT / args.txt_dir
        out = args.output or DEFAULT_GOLDEN_OUTPUT
        out = out if out.is_absolute() else REPO_ROOT / out
        meta = build_merged_from_txt_dir(
            txt_dir,
            out,
            glob=args.glob,
            tenant_id=args.tenant_id,
            align_index=align_index,
            config_dir=str(REPO_ROOT / args.config_dir)
            if not Path(args.config_dir).is_absolute()
            else args.config_dir,
            profile=args.profile,
        )
        print(f"Wrote {meta['count']} samples -> {meta['output']}")
        for name, n in meta["files"].items():
            print(f"  {name}: {n}")
        if meta["id_range"][0]:
            print(f"  ids: {meta['id_range'][0]} .. {meta['id_range'][1]}")
        return 0

    if args.txt:
        txt = args.txt if args.txt.is_absolute() else REPO_ROOT / args.txt
        out = args.output or (
            REPO_ROOT / "data/rag_eval/golden" / f"{txt.stem}.jsonl"
        )
        out = out if out.is_absolute() else REPO_ROOT / out
        if not txt.is_file():
            print(f"TXT not found: {txt}", file=sys.stderr)
            return 1
        config_dir = (
            str(REPO_ROOT / args.config_dir)
            if not Path(args.config_dir).is_absolute()
            else args.config_dir
        )
        rows = build_samples_from_txt(
            txt,
            align_index=align_index,
            config_dir=config_dir,
            profile=args.profile,
        )
        samples = rows_to_dicts(rows)
        for s in samples:
            s["tenant_id"] = args.tenant_id
        write_jsonl(samples, out)
        print(f"Wrote {len(samples)} samples -> {out}")
        return 0

    if args.pdf:
        pdf = args.pdf if args.pdf.is_absolute() else REPO_ROOT / args.pdf
        if not args.output:
            print("PDF mode requires --output (golden dir keeps single merged set)", file=sys.stderr)
            return 1
        out = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        if not pdf.is_file():
            print(f"PDF not found: {pdf}", file=sys.stderr)
            return 1
        meta = build_pdf_dataset(
            pdf, out, tenant_id=args.tenant_id, target_count=args.count
        )
        print(f"Wrote {meta['count']} samples -> {meta['output']}")
        return 0

    # 默认：合并 test_docs txt
    out = args.output or DEFAULT_MERGED_OUTPUT
    out = out if out.is_absolute() else REPO_ROOT / out
    txt_dir = args.txt_dir if args.txt_dir.is_absolute() else REPO_ROOT / args.txt_dir
    config_dir = (
        str(REPO_ROOT / args.config_dir)
        if not Path(args.config_dir).is_absolute()
        else args.config_dir
    )
    meta = build_merged_from_txt_dir(
        txt_dir,
        out,
        glob=args.glob,
        tenant_id=args.tenant_id,
        align_index=align_index,
        config_dir=config_dir,
        profile=args.profile,
    )
    print(f"Wrote {meta['count']} samples -> {meta['output']}")
    for name, n in meta["files"].items():
        print(f"  {name}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
