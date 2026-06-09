# -*- coding: utf-8 -*-
"""直连 ReAct Agent 单轮（与 LangGraph agent 节点同逻辑，供 direct 引擎复用）。"""

from __future__ import annotations

from typing import Any, AsyncIterator, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from core.composition.run_context import RunContext

from app.agents.chat_config import ChatAgentConfig
from app.agents.llm_stream import stream_llm_text
from app.agents.memory_tools import build_memory_tools
from app.runtime.adapters.langgraph.model_bridge import PortChatModel, _message_to_dict


def dict_messages_to_lc(messages: list[dict[str, str]]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def last_ai_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()
    return ""


def _create_agent_compat(model: Any, tools: list) -> Any:
    try:
        from langchain.agents import create_agent

        return create_agent(model, tools=tools or [])
    except Exception:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(model, tools=tools)


def make_react_agent(ctx: RunContext, chat_cfg: ChatAgentConfig) -> Any:
    model = PortChatModel(ctx, model_role=chat_cfg.model_role)
    tools = build_memory_tools(ctx, chat_cfg)
    return _create_agent_compat(model, tools)


async def invoke_direct_llm(
    ctx: RunContext,
    lc_messages: list[BaseMessage],
    chat_cfg: ChatAgentConfig,
) -> str:
    model = PortChatModel(ctx, model_role=chat_cfg.model_role)
    response = await model.ainvoke(lc_messages)
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


async def stream_direct_llm(
    ctx: RunContext,
    lc_messages: list[BaseMessage],
    chat_cfg: ChatAgentConfig,
) -> AsyncIterator[str]:
    """知识类直连 LLM token 流；无 astream 时回退整块输出。"""
    llm = ctx.get_model(chat_cfg.model_role)
    dict_messages = [_message_to_dict(m) for m in lc_messages]
    async for delta in stream_llm_text(llm, dict_messages):
        yield delta


async def invoke_react_agent(
    ctx: RunContext,
    lc_messages: list[BaseMessage],
    chat_cfg: ChatAgentConfig,
) -> str:
    agent = make_react_agent(ctx, chat_cfg)
    result = await agent.ainvoke({"messages": lc_messages})
    return last_ai_text(result.get("messages") or [])
