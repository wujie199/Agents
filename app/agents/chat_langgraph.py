# -*- coding: utf-8 -*-
"""LangGraph 对话入口（无工具 Agent 模式）。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from core.composition.run_context import RunContext

from app.agents.chat_config import ChatAgentConfig, load_chat_config
from app.agents.chat_nodes import fetch_turn_history, persist_user_and_assistant
from app.agents.chat_turn import ChatTurnResult
from app.agents.retrieval_router import should_use_direct_llm_for_intent
from app.agents.react_turn import stream_direct_llm
from app.agents.text_sanitize import strip_model_reasoning
from app.runtime.adapters.langgraph.checkpointer import (
    resolve_chat_checkpointer,
    resolve_chat_checkpointer_async,
)
from app.runtime.adapters.langgraph.engine import LangGraphRuntime
from app.workflows.chat.graph_def import build_chat_langgraph_workflow


@dataclass
class ChatLangGraphSession:
    """一次 REPL 会话持有的已编译图。"""

    compiled: Any
    runtime: LangGraphRuntime
    chat_cfg: ChatAgentConfig


def create_chat_langgraph_session(
    ctx: RunContext,
    *,
    chat_cfg: Optional[ChatAgentConfig] = None,
    checkpointer: Any = None,
) -> ChatLangGraphSession:
    cfg = chat_cfg or load_chat_config()
    cp = checkpointer if checkpointer is not None else resolve_chat_checkpointer(ctx)
    runtime = LangGraphRuntime(checkpointer=cp)
    workflow = build_chat_langgraph_workflow(ctx, cfg)
    compiled = runtime.compile(workflow)
    return ChatLangGraphSession(
        compiled=compiled, runtime=runtime, chat_cfg=cfg
    )


async def create_chat_langgraph_session_async(
    ctx: RunContext,
    *,
    chat_cfg: Optional[ChatAgentConfig] = None,
    checkpointer: Any = None,
) -> ChatLangGraphSession:
    """企业 PG：异步初始化 AsyncPostgresSaver 后编译图。"""
    cfg = chat_cfg or load_chat_config()
    cp = checkpointer
    if cp is None:
        cp = await resolve_chat_checkpointer_async(ctx)
    return create_chat_langgraph_session(ctx, chat_cfg=cfg, checkpointer=cp)


async def run_chat_turn_langgraph(
    session: ChatLangGraphSession,
    ctx: RunContext,
    user_message: str,
    *,
    enable_rag: Optional[bool] = None,
) -> ChatTurnResult:
    """执行一轮 LangGraph 对话（prepare → react_agent → persist）。"""
    from app.agents.memory_runtime_debug import perf_mark

    use_rag = (
        session.chat_cfg.enable_rag
        if enable_rag is None
        else enable_rag
    )
    _turn_t0 = time.perf_counter()
    final = await session.runtime.ainvoke(
        session.compiled,
        {"user_input": user_message.strip()},
        ctx,
        enable_rag=use_rag,
        chat_cfg=session.chat_cfg,
    )
    perf_mark(
        ctx,
        "GRAPH.langgraph_ainvoke",
        (time.perf_counter() - _turn_t0) * 1000,
        enable_rag=use_rag,
    )
    memory = ctx.require_memory()
    history = await fetch_turn_history(
        memory,
        ctx.request,
        max_turns=session.chat_cfg.max_history_turns,
        turn_buffer=ctx.turn_buffer,
    )
    return ChatTurnResult(
        assistant_text=final.get("assistant_text") or "",
        evidence_count=int(final.get("evidence_count") or 0),
        rag_empty=bool(final.get("rag_empty", True)),
        history_turns=len(history),
    )


async def _stream_direct_after_prepare(
    session: ChatLangGraphSession,
    ctx: RunContext,
    *,
    lc_messages: list,
    user_input: str,
    evidence_count: int,
    rag_empty: bool,
) -> AsyncIterator[str]:
    """prepare 完成后直连 LLM 流式 + persist。"""
    yield json.dumps(
        {
            "type": "meta",
            "evidence_count": evidence_count,
            "rag_empty": rag_empty,
            "stream": "direct_llm",
        },
        ensure_ascii=False,
    )
    parts: list[str] = []
    async for delta in stream_direct_llm(ctx, lc_messages, session.chat_cfg):
        parts.append(delta)
        yield json.dumps({"type": "delta", "text": delta}, ensure_ascii=False)
    assistant_text = strip_model_reasoning("".join(parts))
    await persist_user_and_assistant(
        ctx,
        user_message=user_input,
        assistant_text=assistant_text,
        chat_cfg=session.chat_cfg,
    )
    memory = ctx.require_memory()
    history = await fetch_turn_history(
        memory,
        ctx.request,
        max_turns=session.chat_cfg.max_history_turns,
        turn_buffer=ctx.turn_buffer,
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


async def stream_chat_turn_langgraph_events(
    session: ChatLangGraphSession,
    ctx: RunContext,
    user_message: str,
    *,
    enable_rag: Optional[bool] = None,
) -> AsyncIterator[str]:
    """LangGraph SSE：prepare meta + LLM token + done。"""
    use_rag = (
        session.chat_cfg.enable_rag if enable_rag is None else enable_rag
    )
    config = session.runtime.build_config(
        ctx, enable_rag=use_rag, chat_cfg=session.chat_cfg
    )
    input_state = {"user_input": user_message.strip()}
    meta_sent = False
    parts: list[str] = []
    final_state: dict[str, Any] = {}
    direct_handled = False

    try:
        async for event in session.compiled.astream_events(
            input_state, config, version="v2"
        ):
            kind = event.get("event")
            if kind == "on_chain_end" and event.get("name") == "prepare":
                data = event.get("data") or {}
                output = data.get("output") or {}
                intent = str((ctx.extra or {}).get("retrieval_intent") or "")
                if should_use_direct_llm_for_intent(intent, session.chat_cfg):
                    lc_messages = output.get("messages") or []
                    async for payload in _stream_direct_after_prepare(
                        session,
                        ctx,
                        lc_messages=lc_messages,
                        user_input=user_message.strip(),
                        evidence_count=int(output.get("evidence_count") or 0),
                        rag_empty=bool(output.get("rag_empty", True)),
                    ):
                        yield payload
                    direct_handled = True
                    return
                if not meta_sent:
                    yield json.dumps(
                        {
                            "type": "meta",
                            "evidence_count": int(
                                output.get("evidence_count") or 0
                            ),
                            "rag_empty": bool(output.get("rag_empty", True)),
                            "stream": "langgraph",
                        },
                        ensure_ascii=False,
                    )
                    meta_sent = True
            if direct_handled:
                return
            if kind == "on_chat_model_stream":
                chunk = (event.get("data") or {}).get("chunk")
                content = getattr(chunk, "content", None) if chunk else None
                if content:
                    text = str(content)
                    parts.append(text)
                    yield json.dumps(
                        {"type": "delta", "text": text}, ensure_ascii=False
                    )
            if kind == "on_chain_end" and event.get("name") == "LangGraph":
                data = event.get("data") or {}
                final_state = data.get("output") or {}
    except Exception:
        final = await session.runtime.ainvoke(
            session.compiled,
            input_state,
            ctx,
            enable_rag=use_rag,
            chat_cfg=session.chat_cfg,
        )
        if not meta_sent:
            yield json.dumps(
                {
                    "type": "meta",
                    "evidence_count": int(final.get("evidence_count") or 0),
                    "rag_empty": bool(final.get("rag_empty", True)),
                    "stream": "batch",
                },
                ensure_ascii=False,
            )
        text = final.get("assistant_text") or ""
        if text and not parts:
            yield json.dumps({"type": "delta", "text": text}, ensure_ascii=False)
        final_state = final

    assistant_text = (
        final_state.get("assistant_text") or "".join(parts)
    )
    memory = ctx.require_memory()
    history = await fetch_turn_history(
        memory,
        ctx.request,
        max_turns=session.chat_cfg.max_history_turns,
        turn_buffer=ctx.turn_buffer,
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
