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


def offline_rag_data_dir(data_dir: str) -> Path:
    """离线索引数据目录（manifest / bm25 与 chroma 同级）。"""
    _, rag_base = _resolve_dev_rag_paths(data_dir)
    return Path(rag_base)


def manifest_has_tenant(data_dir: str, tenant_id: str) -> bool:
    """检查离线索引清单中是否已有该租户的文档。"""
    from document.rag.application.indexing.index_manifest import IndexManifest

    rag_dir = offline_rag_data_dir(data_dir)
    if not (rag_dir / "indexed_by_md5.json").is_file():
        return False
    return IndexManifest.for_data_dir(rag_dir).has_tenant(tenant_id)


def resolve_rag_tenant_id(
    request: RequestContext,
    *,
    profile: ChatProfile = "dev",
    data_dir: str | None = None,
) -> str:
    """
    RAG 索引 tenant：env 优先；production/dev 默认对齐 memory tenant。
    dev 若该 tenant 已有离线索引记录，始终用 request.tenant_id（与 Web 上传一致）。
    否则在存在旧 default 索引且 RAG_LEGACY_DEFAULT!=false 时回退 default。
    """
    env_tenant = (os.environ.get("RAG_TENANT_ID") or "").strip()
    if env_tenant:
        return env_tenant
    preferred = request.tenant_id
    if profile == "production":
        return preferred
    if preferred == "default":
        return preferred
    if data_dir and manifest_has_tenant(data_dir, preferred):
        return preferred
    legacy_ok = os.environ.get("RAG_LEGACY_DEFAULT", "true").lower() not in (
        "0",
        "false",
        "no",
    )
    if legacy_ok and data_dir:
        chroma_dir, _ = _resolve_dev_rag_paths(data_dir)
        chroma_path = Path(chroma_dir)
        if chroma_path.is_dir() and any(chroma_path.iterdir()):
            return "default"
    return preferred


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
        extra["rag_tenant_id"] = resolve_rag_tenant_id(
            request, profile="production"
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
    extra["rag_tenant_id"] = resolve_rag_tenant_id(
        request, profile="dev", data_dir=data_dir
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


_dev_stack_template: Optional[RunContext] = None


def dev_stack_is_initialized() -> bool:
    return _dev_stack_template is not None


def get_or_build_shared_dev_context(
    request: RequestContext,
    *,
    config_dir: str = "config",
    data_dir: str = "data",
) -> RunContext:
    """dev 进程内复用已装配的 RAG/Memory/Model 栈，仅替换 request（避免重复 ~4s 冷启动）。"""
    global _dev_stack_template
    if _dev_stack_template is None:
        bootstrap = RequestContext(
            tenant_id="__bootstrap__",
            user_id="__bootstrap__",
            session_id="__bootstrap__",
            trace_id="bootstrap",
            channel="system",
        )
        _dev_stack_template = build_chat_run_context(
            bootstrap,
            profile="dev",
            config_dir=config_dir,
            data_dir=data_dir,
        )
    chroma_dir, _ = _resolve_dev_rag_paths(data_dir)
    extra = dict(_dev_stack_template.extra or {})
    extra.setdefault("rag_chroma_dir", chroma_dir)
    extra["rag_tenant_id"] = resolve_rag_tenant_id(
        request, profile="dev", data_dir=data_dir
    )
    extra["data_dir"] = data_dir
    extra["langgraph_checkpoint_path"] = str(
        Path(data_dir) / "langgraph_checkpoints.db"
    )
    return replace(_dev_stack_template, request=request, extra=extra)

