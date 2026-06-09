#!/usr/bin/env python3
"""L4 外部画像同步 cron：HTTP ↔ 本地 YAML 镜像。

用法:
  # 从 HTTP 拉取到本地目录（灾备/离线开发）
  python scripts/l4_profile_sync_cron.py --tenant tenant1 --direction pull

  # 将本地 YAML 推送到 HTTP
  python scripts/l4_profile_sync_cron.py --tenant tenant1 --direction push

crontab 示例:
  0 */6 * * * cd /path/to/Agents && python scripts/l4_profile_sync_cron.py --tenant t1 --direction pull
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_platform.memory.adapters.config_loader import load_memory_config
from agent_platform.memory.adapters.external_factory import build_external_memory
from agent_platform.memory.adapters.file_external_memory_adapter import (
    FileExternalMemoryAdapter,
)
from agent_platform.memory.adapters.http_external_memory_adapter import (
    HttpExternalMemoryAdapter,
)


async def pull_http_to_file(
    tenant_id: str, http: HttpExternalMemoryAdapter, file_adapter: FileExternalMemoryAdapter
) -> dict:
    users = await http.list_profile_users(tenant_id)
    synced = 0
    for user_id in users:
        profile = await http.get_profile(tenant_id, user_id)
        await file_adapter.save_profile(tenant_id, user_id, profile)
        synced += 1
    return {"direction": "pull", "tenant_id": tenant_id, "users_synced": synced}


async def push_file_to_http(
    tenant_id: str, file_adapter: FileExternalMemoryAdapter, http: HttpExternalMemoryAdapter
) -> dict:
    users = await file_adapter.list_profile_users(tenant_id)
    synced = 0
    for user_id in users:
        profile = await file_adapter.get_profile(tenant_id, user_id)
        await http.save_profile(tenant_id, user_id, profile)
        synced += 1
    return {"direction": "push", "tenant_id": tenant_id, "users_synced": synced}


async def main(args: argparse.Namespace) -> int:
    cfg_path = args.config_dir
    if not Path(cfg_path).is_absolute():
        cfg_path = str(REPO_ROOT / cfg_path)
    cfg = load_memory_config(f"{cfg_path}/memory.yml")
    if args.profiles_dir:
        cfg["external_profiles_dir"] = str(args.profiles_dir)

    backend = str(cfg.get("external_profiles_backend", "file")).lower()
    if backend != "http":
        print(
            "sync requires external_profiles_backend=http in memory config; "
            f"got {backend!r}"
        )
        return 1

    http = build_external_memory(cfg)
    while hasattr(http, "_inner"):
        http = http._inner
    if not isinstance(http, HttpExternalMemoryAdapter):
        print("Could not resolve HttpExternalMemoryAdapter from config")
        return 1

    file_adapter = FileExternalMemoryAdapter(
        profiles_dir=cfg.get("external_profiles_dir", "data/external_profiles")
    )

    if args.direction == "pull":
        result = await pull_http_to_file(args.tenant, http, file_adapter)
    else:
        result = await push_file_to_http(args.tenant, file_adapter, http)

    print(
        f"L4 sync {result['direction']}: tenant={result['tenant_id']} "
        f"users={result['users_synced']}"
    )
    return 0


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L4 external profile sync cron")
    p.add_argument("--tenant", required=True)
    p.add_argument(
        "--direction",
        choices=("pull", "push"),
        default="pull",
        help="pull=HTTP→本地YAML, push=本地YAML→HTTP",
    )
    p.add_argument("--config-dir", default="config")
    p.add_argument(
        "--profiles-dir",
        type=Path,
        default=None,
        help="覆盖 external_profiles_dir",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(_parse())))
