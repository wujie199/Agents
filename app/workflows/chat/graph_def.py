# -*- coding: utf-8 -*-
"""
聊天 LangGraph：prepare（L1+RAG+Context）→ create_react_agent → persist（L2）。

Path A: enable_memory_tools=false, tools=[]
Path B: enable_memory_tools=true, L1/L2/L3/L4 记忆工具
"""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage, add_messages

from core.composition.run_context import RunContext

from app.agents.orchestration.chat_config import ChatAgentConfig, load_chat_config
from app.agents.orchestration.chat_nodes import build_turn_messages, persist_user_and_assistant
from app.agents.prompts.text_sanitize import strip_model_reasoning
from app.agents.roles.retrieval_router import should_use_direct_llm_for_intent
from app.agents.roles.react_turn import (
    dict_messages_to_lc,
    invoke_direct_llm,
    last_ai_text,
    make_react_agent,
)
from app.agents.middleware.compose import wrap_node, compose_middlewares
from app.agents.middleware.tracing import TracingMiddleware
from app.agents.middleware.logging import LoggingMiddleware
from app.agents.middleware.policy import PolicyMiddleware
from app.agents.middleware.privacy import PrivacyMiddleware
from app.agents.middleware.audit import AuditMiddleware


class ChatGraphState(TypedDict):
    user_input: str
    messages: Annotated[list, add_messages]
    assistant_text: str
    evidence_count: int
    rag_empty: bool
    memory_snapshot_hash: str
    evidences_summary: list


async def _prepare_node(
    state: ChatGraphState, config: RunnableConfig
) -> dict[str, Any]:
    configurable = (config or {}).get("configurable") or {}
    ctx: RunContext = configurable["run_ctx"]
    _node_t0 = time.perf_counter()
    enable_rag = bool(configurable.get("enable_rag", True))
    chat_cfg: ChatAgentConfig = configurable.get("chat_cfg") or load_chat_config()
    user_input = (state.get("user_input") or "").strip()
    if not user_input:
        for msg in reversed(state.get("messages") or []):
            if isinstance(msg, HumanMessage):
                user_input = str(msg.content or "").strip()
                break

    dict_messages, ev_count, rag_empty, mem_hash, evidences_summary = await build_turn_messages(
        ctx,
        user_input,
        chat_cfg,
        enable_rag=enable_rag,
    )
    lc_messages = dict_messages_to_lc(dict_messages)
    from app.agents.debug.debug_trace import agent_debug

    from app.agents.memory.memory_runtime_debug import perf_mark

    _node_ms = (time.perf_counter() - _node_t0) * 1000
    perf_mark(
        ctx,
        "GRAPH.prepare_node",
        _node_ms,
        evidence_count=ev_count,
        lc_message_count=len(lc_messages),
    )
    agent_debug(
        "GRAPH-PREPARE",
        "graph_def._prepare_node",
        "LangGraph prepare 节点完成",
        {
            "user_input": user_input[:120],
            "enable_rag": enable_rag,
            "evidence_count": ev_count,
            "rag_empty": rag_empty,
            "lc_message_count": len(lc_messages),
            "memory_tools": chat_cfg.enable_memory_tools,
            "duration_ms": round(_node_ms, 2),
        },
    )
    return {
        "user_input": user_input,
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *lc_messages],
        "evidence_count": ev_count,
        "rag_empty": rag_empty,
        "memory_snapshot_hash": mem_hash,
        "evidences_summary": evidences_summary or [],
    }


async def _persist_node(
    state: ChatGraphState, config: RunnableConfig
) -> dict[str, Any]:
    configurable = (config or {}).get("configurable") or {}
    ctx: RunContext = configurable["run_ctx"]
    _node_t0 = time.perf_counter()
    chat_cfg: ChatAgentConfig = configurable.get("chat_cfg") or load_chat_config()
    user_input = state.get("user_input") or ""
    assistant_text = strip_model_reasoning(
        state.get("assistant_text") or last_ai_text(state.get("messages") or [])
    )
    if user_input and assistant_text:
        await persist_user_and_assistant(
            ctx,
            user_message=user_input,
            assistant_text=assistant_text,
            chat_cfg=chat_cfg,
        )
    from app.agents.debug.debug_trace import agent_debug

    from app.agents.memory.memory_runtime_debug import perf_mark

    _node_ms = (time.perf_counter() - _node_t0) * 1000
    perf_mark(ctx, "GRAPH.persist_node", _node_ms)
    agent_debug(
        "GRAPH-PERSIST",
        "graph_def._persist_node",
        "LangGraph persist 节点完成",
        {
            "user_chars": len(user_input or ""),
            "assistant_chars": len(assistant_text or ""),
            "persisted": bool(user_input and assistant_text),
            "duration_ms": round(_node_ms, 2),
        },
    )
    return {"assistant_text": assistant_text}


def _make_sync_compat_wrapped(middlewares, node_fn, node_name):
    """用 middleware 包装节点函数，返回 LangGraph 兼容的异步函数。

    LangGraph 节点签名: async (state, config) -> dict
    wrap_node 返回: async (state, config) -> result
    两者兼容，直接包装即可。
    """
    # wrap_node 是 async 函数返回 wrapped，我们同步调用它来获取 wrapped
    # 但 wrap_node 本身是 async，所以需要 event_loop
    # 改用惰性包装：返回一个 async 函数，首次调用时执行 wrap
    _wrapped = None

    async def _lazy_wrapped(state, config):
        nonlocal _wrapped
        if _wrapped is None:
            _wrapped = await wrap_node(middlewares, node_fn, node_name)
        return await _wrapped(state, config)

    return _lazy_wrapped


def build_chat_langgraph_workflow(
    ctx: RunContext,
    chat_cfg: ChatAgentConfig | None = None,
) -> StateGraph:
    """构建未编译的 StateGraph（外层：prepare → agent → persist）。"""
    cfg = chat_cfg or load_chat_config()
    react_agent = make_react_agent(ctx, cfg)

    # ── 构建 Middleware 链 ──
    middlewares = compose_middlewares(
        TracingMiddleware(),
        PolicyMiddleware(),
        LoggingMiddleware(),
        PrivacyMiddleware(),
        AuditMiddleware(),
    )

    async def _agent_node(
        state: ChatGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        from app.agents.memory.memory_runtime_debug import perf_mark

        _agent_cfg = (config or {}).get("configurable") or {}
        _agent_ctx = _agent_cfg.get("run_ctx")
        _agent_messages = state.get("messages") or []
        intent = ""
        if _agent_ctx is not None and isinstance(
            getattr(_agent_ctx, "extra", None), dict
        ):
            intent = str(_agent_ctx.extra.get("retrieval_intent") or "")
        use_direct = should_use_direct_llm_for_intent(intent, cfg)
        _agent_t0 = time.perf_counter()
        if use_direct:
            assistant_text = strip_model_reasoning(
                await invoke_direct_llm(_agent_ctx, _agent_messages, cfg)
            )
            out_messages = list(_agent_messages)
            if assistant_text:
                from langchain_core.messages import AIMessage

                out_messages = [*out_messages, AIMessage(content=assistant_text)]
        else:
            result = await react_agent.ainvoke(
                {"messages": _agent_messages},
                config,
            )
            out_messages = result.get("messages") or []
            assistant_text = strip_model_reasoning(last_ai_text(out_messages))
        perf_mark(
            _agent_ctx,
            "GRAPH.agent_llm",
            (time.perf_counter() - _agent_t0) * 1000,
            message_count=len(_agent_messages),
            direct_llm=use_direct,
            intent=intent,
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *out_messages,
            ],
            "assistant_text": assistant_text,
        }

    graph = StateGraph(ChatGraphState)
    # 用 middleware 包装关键节点
    _wrapped_prepare = _make_sync_compat_wrapped(middlewares, _prepare_node, "prepare")
    _wrapped_agent = _make_sync_compat_wrapped(middlewares, _agent_node, "agent")
    _wrapped_persist = _make_sync_compat_wrapped(middlewares, _persist_node, "persist")
    graph.add_node("prepare", _wrapped_prepare)
    graph.add_node("agent", _wrapped_agent)
    graph.add_node("persist", _wrapped_persist)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "agent")
    graph.add_edge("agent", "persist")
    graph.add_edge("persist", END)
    return graph
