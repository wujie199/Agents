#!/usr/bin/env python3
"""重放 embedding DLQ：按 doc_id 从 manifest 找到源文件并重新建库。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.ports.index import IndexProfile
from document.build_rag_index import build_one_document
from document.rag.application.indexing.index_manifest import IndexManifest
from document.rag.bootstrap.offline import (
    create_offline_index_service,
    load_offline_config,
)

log = logging.getLogger("replay_embedding_dlq")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_dlq_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            log.warning("跳过无效 DLQ 行: %s", exc)
    return records


def unique_dlq_jobs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 (tenant_id, doc_id) 去重，保留最后一条。"""
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in records:
        tenant_id = str(rec.get("tenant_id") or "default")
        doc_id = str(rec.get("doc_id") or "")
        if not doc_id:
            continue
        seen[(tenant_id, doc_id)] = rec
    return list(seen.values())


async def replay_one(
    rec: Dict[str, Any],
    *,
    data_dir: Path,
    config_dir: str,
    manifest: IndexManifest,
    dry_run: bool,
) -> bool:
    tenant_id = str(rec.get("tenant_id") or "default")
    doc_id = str(rec.get("doc_id") or "")
    found = manifest.find_by_doc_id(tenant_id, doc_id)
    if not found:
        log.error("manifest 未找到 doc_id=%s tenant=%s", doc_id, tenant_id)
        return False

    file_md5, entry = found
    source_path = Path(str(entry.get("source_path") or ""))
    if not source_path.is_file():
        log.error("源文件不存在 doc_id=%s path=%s", doc_id, source_path)
        return False

    if dry_run:
        log.info(
            "DRY-RUN 将重试 doc_id=%s file=%s error=%s",
            doc_id,
            source_path.name,
            rec.get("error"),
        )
        return True

    cfg = load_offline_config(config_dir)
    index_service, chroma_dir = create_offline_index_service(data_dir, cfg)
    report = await build_one_document(
        source_path,
        doc_id,
        tenant_id,
        data_dir,
        config_dir,
        IndexProfile.VECTOR_ONLY,
        cfg=cfg,
        index_service=index_service,
        chroma_dir=chroma_dir,
        manifest=manifest,
        skip_indexed=False,
        force_reindex=True,
    )
    if report.success and not report.errors:
        log.info("重放成功 doc_id=%s file=%s", doc_id, source_path.name)
        return True
    log.error("重放失败 doc_id=%s errors=%s", doc_id, report.errors)
    return False


async def run_replay(
    dlq_path: Path,
    data_dir: Path,
    config_dir: str,
    *,
    dry_run: bool = False,
    limit: int = 0,
) -> int:
    records = load_dlq_records(dlq_path)
    jobs = unique_dlq_jobs(records)
    if limit > 0:
        jobs = jobs[:limit]
    if not jobs:
        log.info("DLQ 无待重放任务: %s", dlq_path)
        return 0

    manifest = IndexManifest.for_data_dir(data_dir)
    ok = 0
    failed = 0
    for rec in jobs:
        if await replay_one(
            rec,
            data_dir=data_dir,
            config_dir=config_dir,
            manifest=manifest,
            dry_run=dry_run,
        ):
            ok += 1
        else:
            failed += 1

    log.info("DLQ 重放完成 成功=%d 失败=%d 总计=%d", ok, failed, len(jobs))
    return 0 if failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="重放 embedding 写入失败 DLQ")
    parser.add_argument(
        "--dlq",
        default=str(REPO_ROOT / "data" / "rag_offline" / "embedding_dlq.jsonl"),
    )
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data" / "rag_offline"),
    )
    parser.add_argument(
        "--config-dir",
        default=str(REPO_ROOT / "config"),
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印将重放的条目")
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 条（0=全部）")
    args = parser.parse_args()

    sys.exit(
        asyncio.run(
            run_replay(
                Path(args.dlq),
                Path(args.data_dir),
                args.config_dir,
                dry_run=args.dry_run,
                limit=args.limit,
            )
        )
    )


if __name__ == "__main__":
    main()
