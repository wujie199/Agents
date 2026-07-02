#!/usr/bin/env python3
"""会话向量 reindex 快捷脚本。

示例:
  MEMORY_PROFILE=vector python scripts/session_vector_reindex.py --tenant tenant1 --user user1

  python scripts/session_vector_reindex.py --tenant tenant1 --session chat1 --batch-size 100
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


async def _run(args: argparse.Namespace) -> int:
    from agent_platform.memory.adapters.config_loader import load_memory_config
    from document.query_memory import _build_memory

    cfg = load_memory_config(
        os.environ.get("MEMORY_CONFIG", "config/memory.yml")
    )
    if not cfg.get("enable_session_vector_index"):
        print(
            "错误: enable_session_vector_index=false。"
            "请设置 MEMORY_PROFILE=vector（或 config/memory.yml 中 enable_session_vector_index: true）",
            file=sys.stderr,
        )
        return 1

    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_vector=True,
    )
    try:
        result = await memory.reindex_session_vectors(
            tenant_id=args.tenant,
            user_id=args.user,
            session_id=args.session,
            batch_size=args.batch_size,
        )
    finally:
        archive = getattr(memory, "_archive_db", None)
        if archive and hasattr(archive, "close"):
            await archive.close()

    if result.get("reason"):
        print(f"Reindex skipped: {result['reason']}")
        return 1
    print(
        f"Reindex done: indexed={result.get('indexed', 0)} "
        f"errors={result.get('errors', 0)} batches={result.get('batches', 0)}"
    )
    return 0 if int(result.get("errors", 0)) == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="会话消息向量 reindex")
    parser.add_argument("--tenant", required=True, help="租户 ID")
    parser.add_argument("--user", default=None, help="可选：限定用户")
    parser.add_argument("--session", default=None, help="可选：限定 session")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
