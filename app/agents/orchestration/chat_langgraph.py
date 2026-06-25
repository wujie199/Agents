# -*- coding: utf-8 -*-
"""LangGraph 对话入口（无工具 Agent 模式）。"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from core.composition.run_context import RunContext

from app.agents.orchestration.chat_config import ChatAgentConfig, load_chat_config
from app.agents.orchestration.chat_nodes import fetch_turn_history, persist_user_and_assistant
from app.agents.orchestration.chat_turn import ChatTurnResult
from app.agents.roles.retrieval_router import should_use_direct_llm_for_intent
from app.agents.roles.react_turn import stream_direct_llm
from app.agents.prompts.text_sanitize import strip_model_reasoning
from app.agents.prompts.llm_stream import stream_llm_chunks
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
    from app.agents.memory.memory_runtime_debug import perf_mark

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
        evidences_summary=final.get("evidences_summary") or [],
    )


async def _stream_direct_after_prepare(
    session: ChatLangGraphSession,
    ctx: RunContext,
    *,
    lc_messages: list,
    user_input: str,
    evidence_count: int,
    rag_empty: bool,
    evidences_summary: list | None = None,
) -> AsyncIterator[str]:
    """prepare 完成后直连 LLM 流式 + persist。"""
    yield json.dumps(
        {
            "type": "meta",
            "evidence_count": evidence_count,
            "rag_empty": rag_empty,
            "evidences_summary": evidences_summary or [],
            "stream": "direct_llm",
        },
        ensure_ascii=False,
    )
    # 先尝试走 stream_llm_chunks 以获取 reasoning
    from app.agents.roles.react_loop import _extract_llm_text

    use_chunk_stream = True
    parts: list[str] = []
    reasoning_parts: list[str] = []
    try:
        llm = ctx.get_model(session.chat_cfg.model_role or "main_llm")
        dict_msgs = []
        for m in lc_messages:
            role = getattr(m, "type", None) or "user"
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            dict_msgs.append({"role": role, "content": str(m.content or "")})
        async for content_delta, reasoning_delta in stream_llm_chunks(llm, dict_msgs):
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
                yield json.dumps(
                    {"type": "thinking", "text": reasoning_delta},
                    ensure_ascii=False,
                )
            if content_delta:
                parts.append(content_delta)
                yield json.dumps(
                    {"type": "delta", "text": content_delta}, ensure_ascii=False
                )
    except Exception:
        # fallback: stream_direct_llm
        use_chunk_stream = False
        async for delta in stream_direct_llm(ctx, lc_messages, session.chat_cfg):
            parts.append(delta)
            yield json.dumps({"type": "delta", "text": delta}, ensure_ascii=False)

    assistant_text = strip_model_reasoning("".join(parts))
    # ── P3: persist 与 history 读取并行化 ──
    _persist_task = asyncio.create_task(
        persist_user_and_assistant(
            ctx,
            user_message=user_input,
            assistant_text=assistant_text,
            chat_cfg=session.chat_cfg,
        )
    )
    await _persist_task
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
    from app.agents.memory.memory_runtime_debug import perf_begin, perf_mark, perf_finish_turn

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

    # ── LLM 流式细粒度计时 ──
    _llm_t0 = time.perf_counter()
    _llm_first_thinking: float | None = None
    _llm_first_delta: float | None = None
    _llm_token_count = 0
    _llm_thinking_chars = 0

    perf_begin(ctx, "stream_langgraph_events")
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
                        evidences_summary=output.get("evidences_summary") or [],
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
                            "evidences_summary": output.get("evidences_summary") or [],
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
                # 提取 reasoning_content（推理模型思考过程）
                additional_kwargs = (
                    getattr(chunk, "additional_kwargs", None) or {}
                    if chunk
                    else {}
                )
                reasoning = additional_kwargs.get("reasoning_content")
                if reasoning:
                    if _llm_first_thinking is None:
                        _llm_first_thinking = time.perf_counter()
                    _llm_thinking_chars += len(str(reasoning))
                    yield json.dumps(
                        {"type": "thinking", "text": str(reasoning)},
                        ensure_ascii=False,
                    )
                if content:
                    text = str(content)
                    if _llm_first_delta is None:
                        _llm_first_delta = time.perf_counter()
                    _llm_token_count += 1
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
                    "evidences_summary": final.get("evidences_summary") or [],
                    "stream": "batch",
                },
                ensure_ascii=False,
            )
        text = final.get("assistant_text") or ""
        if text and not parts:
            yield json.dumps({"type": "delta", "text": text}, ensure_ascii=False)
        final_state = final

    # ── LLM 流式计时汇总 ──
    _llm_wall_ms = (time.perf_counter() - _llm_t0) * 1000
    _ttft_ms = None
    if _llm_first_delta is not None:
        _ttft_ms = (_llm_first_delta - _llm_t0) * 1000
    elif _llm_first_thinking is not None:
        _ttft_ms = (_llm_first_thinking - _llm_t0) * 1000
    _tps = None
    if _llm_first_delta is not None and _llm_token_count > 0:
        _dur = time.perf_counter() - _llm_first_delta
        if _dur > 0:
            _tps = round(_llm_token_count / _dur, 1)

    perf_mark(
        ctx,
        "LLM.stream_wall",
        _llm_wall_ms,
        ttft_ms=round(_ttft_ms, 1) if _ttft_ms is not None else None,
        token_count=_llm_token_count,
        thinking_chars=_llm_thinking_chars,
        tokens_per_sec=_tps,
    )
    if _ttft_ms is not None:
        perf_mark(ctx, "LLM.ttft", _ttft_ms)
    perf_finish_turn(ctx, phase="stream_langgraph_events")

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
