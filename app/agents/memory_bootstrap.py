# -*- coding: utf-8 -*-
"""记忆子系统 dev 启动 bootstrap：开箱即用，无需 MEMORY_CONFIG 切换。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from core.composition.run_context import RunContext
from core.domain.context import RequestContext

from agent_platform.memory.adapters.config_loader import load_memory_config

_logger = logging.getLogger(__name__)

_DEFAULT_L4_YAML = """entities:
  示例用户:
    canonical_id: demo_user
    display_name: 示例用户
facts:
  - key: 部门
    value: 研发部
    source: ldap
  - key: 职位
    value: 工程师
    source: hr
  - key: 语言偏好
    value: 中文
    source: crm
"""


def seed_l4_profile_if_missing(
    *,
    profiles_dir: str | Path,
    tenant_id: str,
    user_id: str,
) -> bool:
    """无 L4 file 画像时写入最小 seed。"""
    root = Path(profiles_dir)
    path = root / tenant_id / f"{user_id}.yaml"
    if path.is_file():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_L4_YAML, encoding="utf-8")
    return True


def ensure_runtime_directories(mem_cfg: dict[str, Any], *, data_dir: str) -> None:
    """创建向量/对象存储/L1 等运行目录。"""
    base = Path(data_dir)
    for key, default in (
        ("store_dir", str(base / "memory_dev")),
        ("session_vector_dir", str(base / "session_vectors")),
    ):
        path = Path(mem_cfg.get(key) or default)
        path.mkdir(parents=True, exist_ok=True)
    (base / "objects").mkdir(parents=True, exist_ok=True)


async def bootstrap_session_vectors(
    memory: Any,
    request: RequestContext,
    mem_cfg: dict[str, Any],
) -> dict[str, Any]:
    """当前会话向量索引补齐（幂等）。"""
    if not mem_cfg.get("enable_session_vector_index"):
        return {"skipped": True, "reason": "vector_index_disabled"}
    reindex = getattr(memory, "reindex_session_vectors", None)
    if reindex is None:
        return {"skipped": True, "reason": "reindex_not_supported"}
    try:
        return await reindex(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            session_id=request.session_id,
            batch_size=int(mem_cfg.get("reindex_batch_size", 200)),
        )
    except Exception as exc:
        _logger.warning("Session vector bootstrap reindex failed: %s", exc)
        return {"skipped": True, "reason": str(exc)}


async def bootstrap_memory_runtime(
    ctx: RunContext,
    *,
    data_dir: str = "data",
    config_dir: str = "config",
    profile: str = "dev",
) -> dict[str, Any]:
    """
    dev 启动一次：目录、L4 seed、ensure_session、会话级向量 reindex。
    production 仅 ensure_session + 目录检查。
    """
    mem_cfg = load_memory_config(f"{config_dir}/memory.yml")
    req = ctx.request
    report: dict[str, Any] = {
        "profile": profile,
        "tenant_id": req.tenant_id,
        "user_id": req.user_id,
        "session_id": req.session_id,
    }

    ensure_runtime_directories(mem_cfg, data_dir=data_dir)

    if str(mem_cfg.get("external_profiles_backend", "file")).lower() == "file":
        seeded = seed_l4_profile_if_missing(
            profiles_dir=mem_cfg.get("external_profiles_dir", f"{data_dir}/external_profiles"),
            tenant_id=req.tenant_id,
            user_id=req.user_id,
        )
        report["l4_profile_seeded"] = seeded

    memory = ctx.memory
    if memory is None:
        report["memory_ready"] = False
        return report

    await memory.ensure_session(req)
    report["memory_ready"] = True

    if profile == "dev":
        report["vector_reindex"] = await bootstrap_session_vectors(
            memory, req, mem_cfg
        )

    if isinstance(getattr(ctx, "extra", None), dict):
        ctx.extra["memory_bootstrap"] = report
    return report
