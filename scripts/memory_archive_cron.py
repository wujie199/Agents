#!/usr/bin/env python3
"""L2 冷归档/过期清理 cron 入口。

用法（crontab 示例）:
  0 3 * * * cd /path/to/Agents && python scripts/memory_archive_cron.py --cold-archive
  0 4 * * 0 cd /path/to/Agents && python scripts/memory_archive_cron.py --cold-archive --backfill-cold-search --tenant t1
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from document.query_memory import _memory_from_args  # noqa: E402


async def main(args: argparse.Namespace) -> int:
    memory = _memory_from_args(
        args,
        enable_cold_archive=args.cold_archive,
        use_vector=args.vector or args.backfill_cold_search,
    )
    exit_code = 0

    if args.cold_archive:
        result = await memory.archive_expired_sessions(retention_days=args.days)
        print(
            f"cold archive: candidates={result.get('candidates', 0)} "
            f"archived={result.get('archived', 0)} "
            f"errors={result.get('errors', 0)}"
        )
        if result.get("errors", 0) > 0:
            exit_code = 1
    else:
        count = await memory.purge_expired_sessions(retention_days=args.days)
        print(f"purged expired rows/sessions: {count}")

    if args.checkpoint_purge:
        from agent_platform.memory.adapters.archive_factory import build_archive_db
        from agent_platform.memory.adapters.config_loader import load_memory_config
        from agent_platform.memory.adapters.relational_checkpointer_adapter import (
            RelationalCheckpointerAdapter,
        )

        mem_cfg = load_memory_config(f"{args.config_dir}/memory.yml")
        if args.pg:
            mem_cfg = {**mem_cfg, "archive_backend": "postgresql"}
        archive = build_archive_db(
            mem_cfg,
            data_dir=str(args.data_dir),
            force_backend="postgresql" if args.pg else None,
        )
        cp = RelationalCheckpointerAdapter(archive)
        days = args.checkpoint_days or mem_cfg.get(
            "checkpoint_retention_days", args.days
        )
        purged = await cp.purge_older_than(int(days))
        print(f"checkpoint purge: days={days} purged={purged}")

    if args.backfill_cold_search:
        if not args.tenant:
            print("backfill-cold-search requires --tenant")
            return 1
        bf = await memory.backfill_cold_search_index(
            tenant_id=args.tenant,
            user_id=args.user,
            limit=args.backfill_limit,
            force=args.force,
        )
        print(
            f"backfill cold search: indexed={bf.get('indexed', 0)} "
            f"skipped={bf.get('skipped', 0)} errors={bf.get('errors', 0)}"
        )
        if bf.get("errors", 0) > 0:
            exit_code = 1

    return exit_code


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L2 archive cron job")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--config-dir", default="config")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--pg", action="store_true")
    p.add_argument(
        "--cold-archive",
        action="store_true",
        help="冷归档（否则硬删除在线明细）",
    )
    p.add_argument("--vector", action="store_true")
    p.add_argument(
        "--backfill-cold-search",
        action="store_true",
        help="归档后补建冷归档 DB 检索索引",
    )
    p.add_argument(
        "--checkpoint-purge",
        action="store_true",
        help="清理超期 graph_checkpoints",
    )
    p.add_argument(
        "--checkpoint-days",
        type=int,
        default=None,
        help="checkpoint 保留天数（默认 checkpoint_retention_days）",
    )
    p.add_argument("--tenant", default=None, help="backfill 目标租户")
    p.add_argument("--user", default=None, help="backfill 目标用户")
    p.add_argument("--backfill-limit", type=int, default=100)
    p.add_argument("--force", action="store_true", help="强制重建冷搜索索引")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(_parse())))
