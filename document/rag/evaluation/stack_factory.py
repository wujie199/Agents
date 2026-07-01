# -*- coding: utf-8 -*-
"""Build evaluation RunContext with RAG cache disabled."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from app.agents.context_factory import build_chat_run_context, resolve_rag_tenant_id

ChatProfile = Literal["dev", "production"]


@dataclass(frozen=True)
class EvalStackConfig:
    generation: dict[str, Any]
    judge: dict[str, Any]
    roles: dict[str, Any]
    metrics: dict[str, list[str]]
    output: dict[str, Any]


def load_eval_config(config_dir: str | Path = "config") -> EvalStackConfig:
    path = Path(config_dir) / "rag_eval.yml"
    if not path.is_file():
        return EvalStackConfig({}, {}, {}, {}, {})
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return EvalStackConfig(
        generation=dict(raw.get("generation") or {}),
        judge=dict(raw.get("judge") or {}),
        roles=dict(raw.get("roles") or {}),
        metrics=dict(raw.get("metrics") or {}),
        output=dict(raw.get("output") or {}),
    )


def _disable_rag_cache(ctx: RunContext) -> None:
    rag = ctx.rag
    if rag is None:
        return
    if hasattr(rag, "_enable_cache"):
        rag._enable_cache = False
    router = getattr(rag, "_router", None)
    if router is not None and hasattr(router, "_enable_cache"):
        router._enable_cache = False
    config = getattr(rag, "_config", None)
    if config is not None and hasattr(config, "enable_cache"):
        try:
            object.__setattr__(config, "enable_cache", False)
        except (AttributeError, TypeError):
            pass


def build_eval_request(
    *,
    sample_tenant_id: str | None,
    default_tenant: str,
    user_id: str,
    session_id: str,
    trace_id: str,
) -> RequestContext:
    tenant = (sample_tenant_id or default_tenant or "default").strip()
    return RequestContext(
        tenant_id=tenant,
        user_id=user_id,
        session_id=session_id,
        trace_id=trace_id,
        channel="rag_eval",
    )


def build_eval_run_context(
    request: RequestContext,
    *,
    profile: ChatProfile = "dev",
    config_dir: str = "config",
    data_dir: str = "data",
    disable_cache: bool = True,
) -> RunContext:
    """Build chat RunContext for evaluation; optionally disable RAG cache."""
    ctx = build_chat_run_context(
        request,
        profile=profile,
        config_dir=config_dir,
        data_dir=data_dir,
    )
    extra = dict(ctx.extra or {})
    extra["rag_tenant_id"] = resolve_rag_tenant_id(
        request, profile=profile, data_dir=data_dir
    )
    extra["data_dir"] = data_dir
    extra["rag_eval_profile"] = profile
    ctx = replace(ctx, extra=extra)
    if disable_cache:
        _disable_rag_cache(ctx)
    return ctx


def resolve_sample_rag_request(
    ctx: RunContext,
    sample_tenant_id: str | None,
    *,
    profile: ChatProfile,
    data_dir: str,
) -> RequestContext:
    """Align retrieval tenant via resolve_rag_tenant_id."""
    base = ctx.request
    if sample_tenant_id:
        base = replace(base, tenant_id=sample_tenant_id.strip())
    rag_tenant = resolve_rag_tenant_id(base, profile=profile, data_dir=data_dir)
    if rag_tenant and rag_tenant != base.tenant_id:
        return replace(base, tenant_id=rag_tenant)
    return base
