#!/usr/bin/env python3
"""将 file/relational L1 热记忆迁移到 LangGraph Store namespace。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_platform.memory.adapters.config_loader import load_memory_config
from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter
from agent_platform.memory.adapters.hot_memory_langgraph_store_adapter import (
    HotMemoryLangGraphStoreAdapter,
    build_langgraph_memory_store,
)


def migrate(store_dir: str, tenant_id: str | None = None) -> int:
    cfg = load_memory_config()
    src = HotMemoryFileAdapter(store_dir=store_dir)
    dst_store = build_langgraph_memory_store(cfg)
    dst = HotMemoryLangGraphStoreAdapter(dst_store)
    base = Path(store_dir)
    tenants = [tenant_id] if tenant_id else [p.name for p in base.iterdir() if p.is_dir()]
    migrated = 0
    for tid in tenants:
        mem = src.get_raw_memory(tid)
        if mem:
            dst.save_memory(tid, mem)
            migrated += 1
        for uid in src.list_user_ids(tid):
            user = src.get_raw_user(tid, uid)
            if user:
                dst.save_user(tid, uid, user)
                migrated += 1
        for uid in src.list_user_ids(tid):
            for delta in src.list_pending_deltas(tid, uid):
                dst.queue_pending_delta(tid, uid, delta)
                migrated += 1
    print(f"Migrated {migrated} documents for tenants: {tenants}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate L1 file store → LangGraph Store")
    parser.add_argument("--store-dir", default="data/memory_dev")
    parser.add_argument("--tenant", default=None)
    args = parser.parse_args()
    return migrate(args.store_dir, args.tenant)


if __name__ == "__main__":
    raise SystemExit(main())
