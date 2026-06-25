# -*- coding: utf-8 -*-
"""单轮对话：记忆 L1/L2 + 可选 RAG + main_llm。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.composition.run_context import RunContext

from agent_platform.memory.adapters.turn_buffer import TurnBuffer

from app.agents.orchestration.chat_config import ChatAgentConfig, load_chat_config
from app.agents.orchestration.chat_nodes import build_turn_messages, persist_user_and_assistant
from app.agents.roles.react_loop import _extract_llm_text, _resolve_turn_buffer
from app.agents.roles.react_turn import (
    dict_messages_to_lc,
    invoke_direct_llm,
    invoke_react_agent,
)
from app.agents.roles.retrieval_router import should_use_direct_llm_for_intent


@dataclass
class ChatTurnResult:
    assistant_text: str
    evidence_count: int = 0
    rag_empty: bool = True
    history_turns: int = 0
    evidences_summary: list = None


async def run_chat_turn(
    ctx: RunContext,
    user_message: str,
    *,
    chat_cfg: Optional[ChatAgentConfig] = None,
    enable_rag: Optional[bool] = None,
    rag_plan: Optional[dict] = None,
    model_role: Optional[str] = None,
    persist: bool = True,
    turn_buffer: Optional[TurnBuffer] = None,
) -> ChatTurnResult:
    """
    执行一轮对话：L1 快照 + 可选 RAG + 历史 turns + LLM，写入 L2。
    """
    memory = ctx.require_memory()
    cfg = chat_cfg or load_chat_config()
    use_rag = cfg.enable_rag if enable_rag is None else enable_rag
    role = model_role or cfg.model_role

    messages, ev_count, rag_empty, _mem_hash, evidences_summary = await build_turn_messages(
        ctx,
        user_message,
        cfg,
        enable_rag=use_rag,
        rag_plan=rag_plan,
    )

    intent = str((ctx.extra or {}).get("retrieval_intent") or "legacy")
    lc_messages = dict_messages_to_lc(messages)

    use_direct = should_use_direct_llm_for_intent(intent, cfg)
    if use_direct:
        assistant_text = await invoke_direct_llm(ctx, lc_messages, cfg)
    elif cfg.enable_memory_tools:
        assistant_text = await invoke_react_agent(ctx, lc_messages, cfg)
    else:
        llm = ctx.get_model(role)
        response = await llm.ainvoke(messages)
        assistant_text = _extract_llm_text(response)

    buf = _resolve_turn_buffer(ctx, turn_buffer)
    if persist:
        await persist_user_and_assistant(
            ctx,
            user_message=user_message,
            assistant_text=assistant_text,
            turn_buffer=buf,
            chat_cfg=cfg,
        )

    from app.agents.orchestration.chat_nodes import fetch_turn_history

    history = await fetch_turn_history(
        memory, ctx.request, max_turns=cfg.max_history_turns, turn_buffer=buf
    )
    return ChatTurnResult(
        assistant_text=assistant_text,
        evidence_count=ev_count,
        rag_empty=rag_empty,
        history_turns=len(history),
        evidences_summary=evidences_summary or [],
    )
