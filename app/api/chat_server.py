# -*- coding: utf-8 -*-
"""HTTP 聊天 API。

启动（需可选依赖）::

    pip install fastapi uvicorn
    uvicorn app.api.chat_server:app --reload --port 8080

或::

    python -m app.api.chat_server --port 8080 --profile production

鉴权（可选）::

    export CHAT_API_KEY=your-secret
    curl -H 'Authorization: Bearer your-secret' ...
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Literal, Optional

from core.domain.context import RequestContext

from agent_platform.infrastructure.observability.adapter import (
    ObservabilityPortAdapter,
)
from agent_platform.infrastructure.observability.otel_adapter import (
    build_observability_port,
)

from app.agents.orchestration.chat_config import (
    ChatAgentConfig,
    load_chat_config,
    load_observability_config,
)
from app.agents.observability.trace_context import resolve_trace_id
from app.agents.orchestration.chat_service import (
    ChatSessionHandle,
    execute_chat_turn,
    format_sse,
    get_l1_snapshot,
    stream_chat_turn_events,
)
from app.agents.memory.memory_views import list_pending_l1_deltas
from app.agents.memory.memory_metrics import (
    get_memory_metric_stats,
    get_memory_metric_stats_from_obs,
    prometheus_text,
)
from app.api.rate_limit import create_chat_rate_limiter
from app.agents.context_factory import ChatProfile, build_chat_run_context
from app.agents.memory.memory_bootstrap import bootstrap_memory_runtime
from app.agents.roles.react_loop import end_agent_session

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import PlainTextResponse, StreamingResponse
    from pydantic import BaseModel, Field

    from app.api.auth import verify_api_key, verify_memory_admin_key
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore[misc, assignment]
    HTTPException = None  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[misc, assignment]
    StreamingResponse = None  # type: ignore[misc, assignment]
    PlainTextResponse = None  # type: ignore[misc, assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    Field = lambda *a, **k: None  # type: ignore[misc, assignment]
    verify_api_key = None  # type: ignore[misc, assignment]
    verify_memory_admin_key = None  # type: ignore[misc, assignment]


class ChatTurnRequest(BaseModel):
    tenant_id: str = "tenant1"
    user_id: str = "user1"
    session_id: str = "api_session"
    message: str
    enable_rag: Optional[bool] = None
    engine: str = Field(default="langgraph", pattern="^(langgraph|direct)$")
    stream_mode: str = Field(default="auto", pattern="^(auto|token|batch)$")
    trace_id: str = "api"


class ChatSnapshotRequest(BaseModel):
    tenant_id: str = "tenant1"
    user_id: str = "user1"
    session_id: str = "api_session"


class ChatEndRequest(BaseModel):
    tenant_id: str = "tenant1"
    user_id: str = "user1"
    session_id: str = "api_session"
    trace_id: str = "api"


class ChatTurnResponse(BaseModel):
    assistant_text: str
    evidence_count: int = 0
    rag_empty: bool = True
    history_turns: int = 0
    session_id: str


class ChatEndResponse(BaseModel):
    ok: bool = True
    session_id: str
    memory_hash: str = ""


@dataclass
class ChatSessionRegistry:
    config_dir: str = "config"
    data_dir: str = "data"
    profile: ChatProfile = "dev"
    enable_memory_tools: bool = True
    observability: ObservabilityPortAdapter = field(
        default_factory=lambda: build_observability_port(service_name="agents-chat")
    )
    _sessions: Dict[str, ChatSessionHandle] = field(default_factory=dict)
    _global_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _key(self, tenant_id: str, user_id: str, session_id: str) -> str:
        return f"{tenant_id}:{user_id}:{session_id}"

    async def get_or_create(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        *,
        engine: str,
    ) -> ChatSessionHandle:
        key = self._key(tenant_id, user_id, session_id)
        async with self._global_lock:
            if key in self._sessions:
                return self._sessions[key]

            chat_cfg = load_chat_config(self.config_dir, profile=self.profile)
            if not self.enable_memory_tools:
                chat_cfg = replace(chat_cfg, enable_memory_tools=False)

            request = RequestContext(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                trace_id="api",
                channel="http",
            )
            run_ctx = build_chat_run_context(
                request,
                profile=self.profile,
                config_dir=self.config_dir,
                data_dir=self.data_dir,
            )
            run_ctx = replace(run_ctx, observability=self.observability)
            await bootstrap_memory_runtime(
                run_ctx,
                data_dir=self.data_dir,
                config_dir=self.config_dir,
                profile=self.profile,
            )

            handle = ChatSessionHandle(run_ctx=run_ctx, chat_cfg=chat_cfg)
            if engine == "langgraph":
                from app.agents.orchestration.chat_langgraph import (
                    create_chat_langgraph_session_async,
                )

                handle.lg_session = await create_chat_langgraph_session_async(
                    run_ctx, chat_cfg=chat_cfg
                )

            self._sessions[key] = handle
            return handle

    async def end(
        self, tenant_id: str, user_id: str, session_id: str
    ) -> Optional[ChatSessionHandle]:
        key = self._key(tenant_id, user_id, session_id)
        async with self._global_lock:
            handle = self._sessions.pop(key, None)
        if handle is None:
            return None
        async with handle.lock:
            await end_agent_session(handle.run_ctx, chat_cfg=handle.chat_cfg)
        return handle


def create_app(
    *,
    config_dir: str = "config",
    data_dir: str = "data",
    profile: ChatProfile = "dev",
) -> Any:
    if FastAPI is None:
        raise RuntimeError(
            "HTTP API 需要安装: pip install fastapi uvicorn"
        )

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        from app.runtime.adapters.langgraph.checkpointer import (
            teardown_postgres_checkpointer,
        )

        await teardown_postgres_checkpointer()

    app = FastAPI(title="Agents Chat API", version="0.2.0", lifespan=lifespan)
    registry = ChatSessionRegistry(
        config_dir=config_dir,
        data_dir=data_dir,
        profile=profile,
    )
    obs_cfg = load_observability_config(config_dir, profile=profile)
    rate_limiter = create_chat_rate_limiter(config_dir)
    auth_dep = [Depends(verify_api_key)] if verify_api_key else []
    memory_admin_dep = (
        [Depends(verify_memory_admin_key)] if verify_memory_admin_key else []
    )
    from app.agents.memory.memory_runtime_debug import set_memory_runtime_debug

    if os.environ.get("MEMORY_RUNTIME_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        set_memory_runtime_debug(True)

    def _enforce_rate_limit(tenant_id: str, user_id: str) -> None:
        allowed, reason = rate_limiter.check(tenant_id, user_id)
        if not allowed:
            raise HTTPException(status_code=429, detail=reason or "请求过于频繁")

    def _apply_http_trace(
        handle: ChatSessionHandle,
        *,
        header_value: Optional[str] = None,
        traceparent: Optional[str] = None,
        body_trace_id: Optional[str] = None,
    ) -> str:
        trace_id = resolve_trace_id(
            header_value=header_value,
            traceparent=traceparent,
            body_trace_id=body_trace_id,
            fallback=getattr(handle.run_ctx.request, "trace_id", None),
        )
        handle.run_ctx = replace(
            handle.run_ctx,
            request=replace(handle.run_ctx.request, trace_id=trace_id),
        )
        return trace_id

    from app.api.health_checks import run_health_checks

    @app.get("/health")
    async def health() -> dict:
        return await run_health_checks(
            config_dir=config_dir,
            data_dir=data_dir,
            profile=profile,
            strict=False,
        )

    @app.get("/ready")
    async def ready() -> dict:
        result = await run_health_checks(
            config_dir=config_dir,
            data_dir=data_dir,
            profile=profile,
            strict=profile == "production",
        )
        if result.get("status") != "healthy":
            raise HTTPException(status_code=503, detail=result)
        return result

    @app.get("/metrics")
    async def metrics_prometheus() -> PlainTextResponse:
        from core.composition.production_factory import build_cache_port

        cache_stats = None
        try:
            cache = build_cache_port(pool_size=2)
            if hasattr(cache, "get_stats"):
                cache_stats = cache.get_stats()
        except Exception:
            pass
        body = prometheus_text(
            observability=registry.observability,
            cache_stats=cache_stats,
        )
        return PlainTextResponse(content=body, media_type="text/plain; version=0.0.4")

    @app.get("/v1/metrics/memory")
    async def metrics_memory() -> dict:
        stats = get_memory_metric_stats_from_obs(registry.observability)
        return {"metrics": stats}

    @app.post("/v1/chat/pending", dependencies=auth_dep)
    async def chat_pending(body: ChatSnapshotRequest) -> dict:
        _enforce_rate_limit(body.tenant_id, body.user_id)
        handle = await registry.get_or_create(
            body.tenant_id,
            body.user_id,
            body.session_id,
            engine="direct",
        )
        return {
            "session_id": body.session_id,
            "pending": list_pending_l1_deltas(handle.run_ctx),
        }

    @app.post("/v1/chat/snapshot", dependencies=auth_dep)
    async def chat_snapshot(body: ChatSnapshotRequest) -> dict:
        _enforce_rate_limit(body.tenant_id, body.user_id)
        handle = await registry.get_or_create(
            body.tenant_id,
            body.user_id,
            body.session_id,
            engine="direct",
        )
        return await get_l1_snapshot(handle)

    @app.post("/v1/chat/turn", response_model=ChatTurnResponse, dependencies=auth_dep)
    async def chat_turn(
        body: ChatTurnRequest,
        request: Request,
    ) -> ChatTurnResponse:
        if not body.message.strip():
            raise HTTPException(status_code=400, detail="message 不能为空")
        _enforce_rate_limit(body.tenant_id, body.user_id)

        handle = await registry.get_or_create(
            body.tenant_id,
            body.user_id,
            body.session_id,
            engine=body.engine,
        )
        if obs_cfg.enabled:
            _apply_http_trace(
                handle,
                header_value=request.headers.get(obs_cfg.trace_header),
                traceparent=request.headers.get("traceparent"),
                body_trace_id=body.trace_id,
            )
        async with handle.lock:
            result = await execute_chat_turn(
                handle,
                body.message,
                engine=body.engine,
                enable_rag=body.enable_rag,
            )
        return ChatTurnResponse(
            assistant_text=result.assistant_text,
            evidence_count=result.evidence_count,
            rag_empty=result.rag_empty,
            history_turns=result.history_turns,
            session_id=body.session_id,
        )

    @app.post("/v1/chat/turn/stream", dependencies=auth_dep)
    async def chat_turn_stream(
        body: ChatTurnRequest,
        request: Request,
    ) -> StreamingResponse:
        if not body.message.strip():
            raise HTTPException(status_code=400, detail="message 不能为空")
        _enforce_rate_limit(body.tenant_id, body.user_id)

        handle = await registry.get_or_create(
            body.tenant_id,
            body.user_id,
            body.session_id,
            engine=body.engine,
        )
        if obs_cfg.enabled:
            _apply_http_trace(
                handle,
                header_value=request.headers.get(obs_cfg.trace_header),
                traceparent=request.headers.get("traceparent"),
                body_trace_id=body.trace_id,
            )

        async def event_stream():
            async with handle.lock:
                async for payload in stream_chat_turn_events(
                    handle,
                    body.message,
                    engine=body.engine,
                    enable_rag=body.enable_rag,
                    stream_mode=body.stream_mode,  # type: ignore[arg-type]
                ):
                    yield format_sse(payload)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/chat/end", response_model=ChatEndResponse, dependencies=auth_dep)
    async def chat_end(body: ChatEndRequest) -> ChatEndResponse:
        handle = await registry.end(
            body.tenant_id, body.user_id, body.session_id
        )
        if handle is None:
            raise HTTPException(status_code=404, detail="session 不存在")
        snap = handle.run_ctx.require_memory().compose_prompt_snapshot(
            handle.run_ctx.request
        )
        return ChatEndResponse(
            session_id=body.session_id,
            memory_hash=snap.hash,
        )

    async def _get_handle(tenant_id: str, user_id: str, session_id: str):
        return await registry.get_or_create(
            tenant_id, user_id, session_id, engine="direct"
        )

    from app.api.memory_routes import create_memory_router

    memory_router = create_memory_router(
        get_handle=_get_handle,
        enforce_rate_limit=_enforce_rate_limit,
        auth_dep=auth_dep,
        memory_admin_dep=memory_admin_dep,
    )
    app.include_router(memory_router)

    # ── RAG 知识库路由 ──
    try:
        from app.api.rag_routes import create_rag_router

        rag_router = create_rag_router(
            config_dir=config_dir,
            data_dir=data_dir,
            enforce_rate_limit=_enforce_rate_limit,
            auth_dep=auth_dep,
        )
        app.include_router(rag_router)
    except Exception as exc:
        import logging
        logging.getLogger("chat_server").warning("RAG 路由注册失败（可选）: %s", exc)

    return app


try:
    app = create_app()
except RuntimeError:
    app = None  # type: ignore[misc, assignment]


def main() -> int:
    parser = argparse.ArgumentParser(description="Agents HTTP Chat API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--profile",
        choices=("dev", "production"),
        default="dev",
        help="RunContext 配置：dev 或 production",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("请安装: pip install fastapi uvicorn")
        return 1

    global app
    app = create_app(
        config_dir=args.config_dir,
        data_dir=args.data_dir,
        profile=args.profile,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
