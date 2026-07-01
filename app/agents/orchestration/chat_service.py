# -*- coding: utf-8 -*-
"""聊天服务层：统一 turn 执行与 SSE 事件生成。"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Optional

from core.composition.run_context import RunContext

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.orchestration.chat_nodes import build_turn_messages, persist_user_and_assistant
from app.agents.orchestration.chat_turn import ChatTurnResult, run_chat_turn
from app.agents.orchestration.chat_langgraph import (
    ChatLangGraphSession,
    create_chat_langgraph_session,
    create_chat_langgraph_session_async,
    run_chat_turn_langgraph,
    stream_chat_turn_langgraph_events,
)
from app.agents.prompts.llm_stream import stream_llm_text, stream_llm_chunks
from app.agents.prompts.text_sanitize import strip_model_reasoning
from app.agents.memory.memory_metrics import (
    record_chat_turn,
    record_rag_cache_stats,
    record_redis_cache_stats,
)
from app.agents.memory.memory_runtime_debug import (
    is_memory_runtime_debug,
    log_memory_runtime_status,
    log_turn_trace,
    perf_begin,
    perf_finish_turn,
    perf_mark,
)
from app.agents.roles.react_loop import _resolve_turn_buffer

StreamMode = Literal["auto", "token", "batch"]


@dataclass
class ChatSessionHandle:
    run_ctx: RunContext
    chat_cfg: ChatAgentConfig
    lg_session: Optional[ChatLangGraphSession] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def execute_chat_turn(
    handle: ChatSessionHandle,
    message: str,
    *,
    engine: str = "langgraph",
    enable_rag: Optional[bool] = None,
) -> ChatTurnResult:
    text = (message or "").strip()
    if not text:
        raise ValueError("message 不能为空")

    use_rag = (
        handle.chat_cfg.enable_rag if enable_rag is None else enable_rag
    )

    if is_memory_runtime_debug():
        await log_turn_trace(
            handle.run_ctx,
            phase="turn_start",
            user_message=text,
            extra={"engine": engine, "enable_rag": use_rag},
        )

    perf_begin(handle.run_ctx, "execute_chat_turn")
    _exec_t0 = time.perf_counter()

    if engine == "langgraph":
        if handle.lg_session is None:
            handle.lg_session = await create_chat_langgraph_session_async(
                handle.run_ctx, chat_cfg=handle.chat_cfg
            )
        result = await run_chat_turn_langgraph(
            handle.lg_session,
            handle.run_ctx,
            text,
            enable_rag=use_rag,
        )
    else:
        result = await run_chat_turn(
            handle.run_ctx,
            text,
            chat_cfg=handle.chat_cfg,
            enable_rag=use_rag,
        )

    perf_mark(
        handle.run_ctx,
        "execute_chat_turn_wall",
        (time.perf_counter() - _exec_t0) * 1000,
        engine=engine,
        evidence_count=result.evidence_count,
    )
    perf_finish_turn(
        handle.run_ctx,
        phase="execute_chat_turn",
        extra={
            "engine": engine,
            "assistant_chars": len(result.assistant_text or ""),
        },
    )

    record_chat_turn(
        handle.run_ctx,
        evidence_count=result.evidence_count,
        rag_empty=result.rag_empty,
        history_turns=result.history_turns,
        engine=engine,
    )
    record_rag_cache_stats(handle.run_ctx)
    record_redis_cache_stats(handle.run_ctx)
    if is_memory_runtime_debug():
        extra_trace = {
            "engine": engine,
            "evidence_count": result.evidence_count,
            "rag_empty": result.rag_empty,
            "history_turns": result.history_turns,
            "assistant_chars": len(result.assistant_text or ""),
            "assistant_preview": (result.assistant_text or "")[:400],
        }
        decision = (handle.run_ctx.extra or {}).get("turn_decision")
        if isinstance(decision, dict):
            extra_trace["decision"] = decision
        await log_turn_trace(
            handle.run_ctx,
            phase="turn_done",
            user_message=text,
            extra=extra_trace,
        )
    return result


def _chunk_text(text: str, size: int = 48) -> list[str]:
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def _use_token_stream(
    handle: ChatSessionHandle,
    *,
    engine: str,
    stream_mode: StreamMode,
) -> bool:
    if stream_mode == "batch":
        return False
    if stream_mode == "token":
        return engine == "direct" and not handle.chat_cfg.enable_memory_tools
    return engine == "direct" and not handle.chat_cfg.enable_memory_tools


async def _stream_direct_token_turn(
    handle: ChatSessionHandle,
    message: str,
    *,
    enable_rag: Optional[bool] = None,
) -> AsyncIterator[str]:
    """direct + 无工具：prepare 后 token 级流式 LLM，再 persist。"""
    ctx = handle.run_ctx
    cfg = handle.chat_cfg
    use_rag = cfg.enable_rag if enable_rag is None else enable_rag

    messages, ev_count, rag_empty, _mem_hash, evidences_summary, memory_summary = await build_turn_messages(
        ctx,
        message,
        cfg,
        enable_rag=use_rag,
    )
    meta = {
        "type": "meta",
        "evidence_count": ev_count,
        "rag_empty": rag_empty,
        "evidences_summary": evidences_summary or [],
        "memory_summary": memory_summary or {},
        "stream": "token",
    }
    yield json.dumps(meta, ensure_ascii=False)

    llm = ctx.get_model(cfg.model_role)
    parts: list[str] = []
    async for content_delta, reasoning_delta in stream_llm_chunks(llm, messages):
        if reasoning_delta:
            yield json.dumps(
                {"type": "thinking", "text": reasoning_delta}, ensure_ascii=False
            )
        if content_delta:
            parts.append(content_delta)
            yield json.dumps(
                {"type": "delta", "text": content_delta}, ensure_ascii=False
            )

    assistant_text = strip_model_reasoning("".join(parts))
    buf = _resolve_turn_buffer(ctx, None)
    await persist_user_and_assistant(
        ctx,
        user_message=message,
        assistant_text=assistant_text,
        turn_buffer=buf,
        chat_cfg=cfg,
    )

    from app.agents.orchestration.chat_nodes import fetch_turn_history

    history = await fetch_turn_history(
        ctx.require_memory(),
        ctx.request,
        max_turns=cfg.max_history_turns,
        turn_buffer=buf,
    )
    yield json.dumps(
        {
            "type": "done",
            "assistant_text": assistant_text,
            "session_id": ctx.request.session_id,
            "history_turns": len(history),
        },
        ensure_ascii=False,
    )


async def stream_chat_turn_events(
    handle: ChatSessionHandle,
    message: str,
    *,
    engine: str = "langgraph",
    enable_rag: Optional[bool] = None,
    chunk_size: int = 48,
    stream_mode: StreamMode = "auto",
) -> AsyncIterator[str]:
    """生成 SSE JSON payload（调用方包装为 SSE）。"""
    perf_begin(handle.run_ctx, "stream_chat_turn_events")
    _stream_t0 = time.perf_counter()
    try:
        if engine == "langgraph" and stream_mode in ("auto", "token"):
            if handle.lg_session is None:
                handle.lg_session = await create_chat_langgraph_session_async(
                    handle.run_ctx, chat_cfg=handle.chat_cfg
                )
            async for payload in stream_chat_turn_langgraph_events(
                handle.lg_session,
                handle.run_ctx,
                message,
                enable_rag=enable_rag,
            ):
                yield payload
            return

        if _use_token_stream(handle, engine=engine, stream_mode=stream_mode):
            async for payload in _stream_direct_token_turn(
                handle, message, enable_rag=enable_rag
            ):
                yield payload
            return

        result = await execute_chat_turn(
            handle,
            message,
            engine=engine,
            enable_rag=enable_rag,
        )
        meta = {
            "type": "meta",
            "evidence_count": result.evidence_count,
            "rag_empty": result.rag_empty,
            "evidences_summary": result.evidences_summary or [],
            "memory_summary": result.memory_summary or {},
            "history_turns": result.history_turns,
            "stream": "batch",
        }
        yield json.dumps(meta, ensure_ascii=False)

        for piece in _chunk_text(result.assistant_text, chunk_size):
            yield json.dumps({"type": "delta", "text": piece}, ensure_ascii=False)

        yield json.dumps(
            {
                "type": "done",
                "assistant_text": result.assistant_text,
                "session_id": handle.run_ctx.request.session_id,
            },
            ensure_ascii=False,
        )
    finally:
        perf_mark(
            handle.run_ctx,
            "stream_chat_turn_events_wall",
            (time.perf_counter() - _stream_t0) * 1000,
            engine=engine,
        )
        perf_finish_turn(
            handle.run_ctx,
            phase="stream_chat_turn_events",
            extra={"engine": engine},
        )


def format_sse(data: str) -> str:
    return f"data: {data}\n\n"


async def get_l1_snapshot(handle: ChatSessionHandle) -> dict:
    snap = handle.run_ctx.require_memory().compose_prompt_snapshot(
        handle.run_ctx.request
    )
    return {
        "hash": snap.hash,
        "memory_text": snap.memory_text,
        "session_id": handle.run_ctx.request.session_id,
    }
