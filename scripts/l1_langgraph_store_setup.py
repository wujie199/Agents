#!/usr/bin/env python3
"""初始化 LangGraph PostgresStore（L1 langgraph 后端建表）。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_platform.memory.adapters.l1_langgraph_store_registry import (
    setup_postgres_memory_store,
    teardown_postgres_memory_store,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup LangGraph PostgresStore for L1")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", ""),
        help="PostgreSQL DSN (default: DATABASE_URL)",
    )
    args = parser.parse_args()
    if not args.database_url:
        print("ERROR: set DATABASE_URL or pass --database-url", file=sys.stderr)
        return 1
    store = setup_postgres_memory_store(args.database_url)
    ns = ("memory", "_healthcheck")
    store.put(ns, "ping", {"content": "ok"})
    item = store.get(ns, "ping")
    print(f"PostgresStore setup OK: {item}")
    teardown_postgres_memory_store()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
