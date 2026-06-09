# -*- coding: utf-8 -*-
"""聊天场景 RunContext 工厂（dev / production）。"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Literal, Optional

from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from agent_platform.memory.adapters.config_loader import (
    ensure_memory_config_env,
    load_memory_config,
)
from core.composition.production_factory import (
    build_development_context,
    build_production_context,
    _resolve_dev_rag_paths,
)

ChatProfile = Literal["dev", "production"]


def resolve_rag_tenant_id(
    request: RequestContext,
    *,
    profile: ChatProfile = "dev",
) -> str:
    """RAG 索引 tenant：env 优先；production 对齐 memory tenant；dev 默认 default（兼容现有 chroma）。"""
    env_tenant = (os.environ.get("RAG_TENANT_ID") or "").strip()
    if env_tenant:
        return env_tenant
    if profile == "production":
        return request.tenant_id
    return "default"


def build_chat_run_context(
    request: RequestContext,
    *,
    profile: ChatProfile = "dev",
    config_dir: str = "config",
    data_dir: str = "data",
) -> RunContext:
    """按 profile 构建聊天 RunContext。"""
    if profile == "production":
        ensure_memory_config_env(config_dir, profile="production")
        ctx = build_production_context(
            request,
            config_dir=config_dir,
            data_dir=data_dir,
            use_memory_graph=_env_bool("USE_MEMORY_GRAPH", False),
        )
        extra = dict(ctx.extra or {})
        extra.setdefault(
            "rag_tenant_id",
            resolve_rag_tenant_id(request, profile="production"),
        )
        extra.setdefault("rag_chroma_dir", f"{data_dir}/chroma")
        extra["data_dir"] = data_dir
        extra["langgraph_checkpoint_path"] = os.environ.get(
            "LANGGRAPH_CHECKPOINT_PATH",
            str(Path(data_dir) / "langgraph_checkpoints.db"),
        )
        mem_cfg = load_memory_config()
        extra["memory_config_summary"] = {
            "memory_config_path": mem_cfg.get("_config_path"),
            "archive_backend": mem_cfg.get("archive_backend", "sqlite"),
            "l1_store_backend": mem_cfg.get("l1_store_backend", "file"),
            "store_dir": mem_cfg.get("store_dir"),
            "enable_cold_archive": mem_cfg.get("enable_cold_archive"),
            "enable_session_vector_index": mem_cfg.get(
                "enable_session_vector_index"
            ),
        }
        return replace(ctx, extra=extra)

    ctx = build_development_context(
        request,
        config_dir=config_dir,
        data_dir=data_dir,
    )
    chroma_dir, _ = _resolve_dev_rag_paths(data_dir)
    extra = dict(ctx.extra or {})
    extra.setdefault("rag_chroma_dir", chroma_dir)
    extra.setdefault(
        "rag_tenant_id",
        resolve_rag_tenant_id(request, profile="dev"),
    )
    extra["data_dir"] = data_dir
    extra["langgraph_checkpoint_path"] = str(
        Path(data_dir) / "langgraph_checkpoints.db"
    )
    return replace(ctx, extra=extra)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

