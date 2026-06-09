# -*- coding: utf-8 -*-
"""会话结束：L2→L1 抽取 + finalize。"""

from __future__ import annotations

from typing import Any, List

from core.composition.run_context import RunContext
from core.ports.memory import MemoryDelta

from app.agents.chat_config import ChatAgentConfig, load_chat_config
from app.agents.context_builder import extract_l1_facts_from_session, validate_l1_key, is_allowed_l1_value
from app.agents.chat_nodes import fetch_turn_history
from app.agents.debug_trace import agent_debug


async def enrich_l1_before_finalize(
    ctx: RunContext,
    chat_cfg: ChatAgentConfig | None = None,
) -> int:
    """finalize 前：从 L2 抽取 KV 写入 pending L1。返回 pending 条数。"""
    cfg = chat_cfg or load_chat_config()
    if not cfg.enable_l1_extract_on_finalize:
        return 0

    memory = ctx.require_memory()
    turns = await fetch_turn_history(
        memory,
        ctx.request,
        max_turns=cfg.l1_extract_max_turns,
        turn_buffer=ctx.turn_buffer,
    )
    facts = await extract_l1_facts_from_session(ctx, turns, cfg)
    written = 0
    for fact in facts:
        try:
            key = validate_l1_key(fact["key"], cfg.l1_extract_allowed_keys)
        except ValueError:
            continue
        val = (fact.get("value") or "").strip()
        if not val:
            continue
        if not is_allowed_l1_value(key, val):
            continue
        await memory.update_prompt_memory(
            ctx.request,
            MemoryDelta(key=key, value=val, source="user"),
            require_hitl=cfg.remember_require_hitl,
        )
        written += 1
    agent_debug(
        "FINALIZE-L1",
        "session_finalize.enrich_l1_before_finalize",
        "L2→L1 抽取完成",
        {
            "turns_scanned": len(turns),
            "facts_extracted": len(facts),
            "facts_written": written,
            "hitl": cfg.remember_require_hitl,
        },
    )
    return written
