# -*- coding: utf-8 -*-
"""企业级记忆管理 HTTP 路由（挂载到 Chat API）。"""

from __future__ import annotations

from typing import Any, Optional

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore[misc, assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    Field = lambda *a, **k: None  # type: ignore[misc, assignment]

from app.agents.memory.enterprise_memory import (
    confirm_pending_l1,
    get_memory_status,
    list_user_sessions_enriched,
    memory_config_summary,
    refresh_l4_profile,
    purge_old_checkpoints,
    purge_tenant_l3_memory,
    purge_tenant_l4_memory,
    purge_user_memory,
    run_retention_cleanup,
)
from app.agents.memory.memory_views import list_pending_l1_deltas


class TenantUserRequest(BaseModel):
    tenant_id: str = "tenant1"
    user_id: str = "user1"
    session_id: str = "admin"


class SessionListRequest(TenantUserRequest):
    limit: int = Field(default=20, ge=1, le=100)


class PurgeUserRequest(TenantUserRequest):
    confirm: bool = False


class RetentionRequest(BaseModel):
    retention_days: Optional[int] = Field(default=None, ge=1, le=3650)


class PurgeTenantRequest(BaseModel):
    tenant_id: str
    confirm: bool = False
    delete_runs: bool = True


def register_memory_routes(
    router: Any,
    *,
    get_handle,
    enforce_rate_limit,
    memory_admin_dep=None,
) -> None:
    """注册 /v1/memory/* 路由。"""

    @router.post("/v1/memory/status")
    async def memory_status(body: TenantUserRequest) -> dict:
        enforce_rate_limit(body.tenant_id, body.user_id)
        handle = await get_handle(body.tenant_id, body.user_id, body.session_id)
        return await get_memory_status(handle.run_ctx)

    @router.post("/v1/memory/sessions")
    async def memory_sessions(body: SessionListRequest) -> dict:
        enforce_rate_limit(body.tenant_id, body.user_id)
        handle = await get_handle(body.tenant_id, body.user_id, body.session_id)
        rows = await list_user_sessions_enriched(handle.run_ctx, limit=body.limit)
        return {
            "tenant_id": body.tenant_id,
            "user_id": body.user_id,
            "sessions": rows,
            "count": len(rows),
        }

    @router.post("/v1/memory/pending")
    async def memory_pending(body: TenantUserRequest) -> dict:
        enforce_rate_limit(body.tenant_id, body.user_id)
        handle = await get_handle(body.tenant_id, body.user_id, body.session_id)
        pending = list_pending_l1_deltas(handle.run_ctx)
        return {
            "tenant_id": body.tenant_id,
            "user_id": body.user_id,
            "pending": pending,
        }

    @router.post("/v1/memory/confirm")
    async def memory_confirm(body: TenantUserRequest) -> dict:
        enforce_rate_limit(body.tenant_id, body.user_id)
        handle = await get_handle(body.tenant_id, body.user_id, body.session_id)
        n = await confirm_pending_l1(handle.run_ctx)
        return {"confirmed": n}

    @router.post("/v1/memory/l4/refresh")
    async def memory_l4_refresh(body: TenantUserRequest) -> dict:
        enforce_rate_limit(body.tenant_id, body.user_id)
        handle = await get_handle(body.tenant_id, body.user_id, body.session_id)
        return await refresh_l4_profile(handle.run_ctx)

    admin_deps = memory_admin_dep or []

    @router.post("/v1/memory/purge/user", dependencies=admin_deps)
    async def memory_purge_user(body: PurgeUserRequest) -> dict:
        if not body.confirm:
            raise HTTPException(
                status_code=400,
                detail="purge 需 confirm=true（不可逆）",
            )
        enforce_rate_limit(body.tenant_id, body.user_id)
        handle = await get_handle(body.tenant_id, body.user_id, body.session_id)
        result = await purge_user_memory(
            handle.run_ctx,
            tenant_id=body.tenant_id,
            user_id=body.user_id,
        )
        return {"ok": True, "result": result}

    @router.post("/v1/memory/retention/run", dependencies=admin_deps)
    async def memory_retention(body: RetentionRequest) -> dict:
        handle = await get_handle("system", "admin", "retention")
        count = await run_retention_cleanup(
            handle.run_ctx, retention_days=body.retention_days
        )
        return {"purged": count}

    @router.post("/v1/memory/purge/tenant/l3", dependencies=admin_deps)
    async def memory_purge_tenant_l3(body: PurgeTenantRequest) -> dict:
        if not body.confirm:
            raise HTTPException(
                status_code=400,
                detail="purge 需 confirm=true（不可逆）",
            )
        handle = await get_handle("system", "admin", f"purge_l3_{body.tenant_id}")
        result = await purge_tenant_l3_memory(
            handle.run_ctx,
            tenant_id=body.tenant_id,
            delete_runs=body.delete_runs,
        )
        return {"ok": True, "result": result}

    @router.post("/v1/memory/purge/tenant/l4", dependencies=admin_deps)
    async def memory_purge_tenant_l4(body: PurgeTenantRequest) -> dict:
        if not body.confirm:
            raise HTTPException(
                status_code=400,
                detail="purge 需 confirm=true（不可逆）",
            )
        handle = await get_handle("system", "admin", f"purge_l4_{body.tenant_id}")
        result = await purge_tenant_l4_memory(
            handle.run_ctx, tenant_id=body.tenant_id
        )
        return {"ok": True, "result": result}

    @router.post("/v1/memory/checkpoint/purge", dependencies=admin_deps)
    async def memory_checkpoint_purge(body: RetentionRequest) -> dict:
        handle = await get_handle("system", "admin", "checkpoint_purge")
        count = await purge_old_checkpoints(
            handle.run_ctx, retention_days=body.retention_days
        )
        return {"purged": count}

    @router.get("/v1/memory/runtime")
    async def memory_runtime() -> dict:
        handle = await get_handle("system", "admin", "runtime")
        from app.agents.memory.memory_runtime_debug import collect_memory_runtime_status

        return await collect_memory_runtime_status(
            handle.run_ctx, event="http_runtime"
        )

    @router.get("/v1/memory/config")
    async def memory_config() -> dict:
        return memory_config_summary()


def create_memory_router(
    *,
    get_handle,
    enforce_rate_limit,
    auth_dep=None,
    memory_admin_dep=None,
) -> Any:
    if APIRouter is None:
        raise RuntimeError("需要 fastapi")
    router = APIRouter(
        dependencies=auth_dep or [],
    )
    register_memory_routes(
        router,
        get_handle=get_handle,
        enforce_rate_limit=enforce_rate_limit,
        memory_admin_dep=memory_admin_dep,
    )
    return router
