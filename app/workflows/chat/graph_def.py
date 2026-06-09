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

from app.agents.chat_config import ChatAgentConfig, load_chat_config
from app.agents.chat_nodes import build_turn_messages, persist_user_and_assistant
from app.agents.text_sanitize import strip_model_reasoning
from app.agents.retrieval_router import should_use_direct_llm_for_intent
from app.agents.react_turn import (
    dict_messages_to_lc,
    invoke_direct_llm,
    last_ai_text,
    make_react_agent,
)


class ChatGraphState(TypedDict):
    user_input: str
    messages: Annotated[list, add_messages]
    assistant_text: str
    evidence_count: int
    rag_empty: bool
    memory_snapshot_hash: str


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

    dict_messages, ev_count, rag_empty, mem_hash = await build_turn_messages(
        ctx,
        user_input,
        chat_cfg,
        enable_rag=enable_rag,
    )
    lc_messages = dict_messages_to_lc(dict_messages)
    from app.agents.debug_trace import agent_debug

    from app.agents.memory_runtime_debug import perf_mark

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
    from app.agents.debug_trace import agent_debug

    from app.agents.memory_runtime_debug import perf_mark

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


def build_chat_langgraph_workflow(
    ctx: RunContext,
    chat_cfg: ChatAgentConfig | None = None,
) -> StateGraph:
    """构建未编译的 StateGraph（外层：prepare → agent → persist）。"""
    cfg = chat_cfg or load_chat_config()
    react_agent = make_react_agent(ctx, cfg)

    async def _agent_node(
        state: ChatGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        from app.agents.memory_runtime_debug import perf_mark

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
    graph.add_node("prepare", _prepare_node)
    graph.add_node("agent", _agent_node)
    graph.add_node("persist", _persist_node)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "agent")
    graph.add_edge("agent", "persist")
    graph.add_edge("persist", END)
    return graph
