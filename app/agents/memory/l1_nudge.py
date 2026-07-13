# -*- coding: utf-8 -*-
"""L1 后台记忆回顾 nudge：周期性从 L2 抽取偏好写入 pending L1。"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.composition.run_context import RunContext
from core.ports.memory import MemoryDelta

from app.agents.context_builder import (
    extract_l1_facts_from_session,
    is_allowed_l1_value,
    validate_l1_key,
)
from app.agents.memory.memory_graph_state import append_pending_memory_delta
from app.agents.orchestration.chat_config import ChatAgentConfig, load_chat_config
from app.agents.orchestration.chat_nodes import fetch_turn_history

logger = logging.getLogger(__name__)

_NUDGE_TURN_KEY = "_l1_nudge_user_turns"
_LAST_NUDGE_SCAN_KEY = "_l1_nudge_last_scan_turns"


def _extra(ctx: RunContext) -> dict:
    extra = getattr(ctx, "extra", None)
    return extra if isinstance(extra, dict) else {}


def record_user_turn_for_nudge(ctx: RunContext) -> int:
    extra = _extra(ctx)
    count = int(extra.get(_NUDGE_TURN_KEY, 0)) + 1
    extra[_NUDGE_TURN_KEY] = count
    return count


async def maybe_nudge_memory_review(
    ctx: RunContext,
    *,
    nudge_interval: int = 10,
    summarizer: Any = None,
    chat_cfg: Optional[ChatAgentConfig] = None,
) -> Optional[dict[str, Any]]:
    """每 nudge_interval 个 user turn 从 L2 抽取 KV 写入 pending L1。"""
    if nudge_interval <= 0:
        return None
    turns = record_user_turn_for_nudge(ctx)
    if turns % nudge_interval != 0:
        return None

    cfg = chat_cfg or load_chat_config()
    if not cfg.enable_memory_tools:
        return {"skipped": True, "reason": "memory_tools_disabled", "user_turns": turns}
    if not cfg.enable_l1_extract_on_finalize and not cfg.use_llm_l1_extract:
        return {"skipped": True, "reason": "l1_extract_disabled", "user_turns": turns}
    if ctx.models is None:
        return {"skipped": True, "reason": "no_models", "user_turns": turns}

    extra = _extra(ctx)
    last_scan = int(extra.get(_LAST_NUDGE_SCAN_KEY) or 0)
    extra[_LAST_NUDGE_SCAN_KEY] = turns

    memory = ctx.require_memory()
    history = await fetch_turn_history(
        memory,
        ctx.request,
        max_turns=cfg.l1_extract_max_turns,
        turn_buffer=ctx.turn_buffer,
    )
    if last_scan > 0 and len(history) > last_scan * 2:
        history = history[-(turns - last_scan) * 2 :]

    facts = await extract_l1_facts_from_session(ctx, history, cfg)
    if not facts:
        logger.debug(
            "L1 nudge: no facts session=%s turn=%s",
            ctx.request.session_id,
            turns,
        )
        return {"nudged": True, "user_turns": turns, "facts_written": 0}

    min_conf = float(cfg.l1_auto_write_confidence_min)
    written = 0
    for fact in facts:
        try:
            key = validate_l1_key(fact["key"], cfg.l1_extract_allowed_keys)
        except ValueError:
            continue
        val = (fact.get("value") or "").strip()
        if not val or not is_allowed_l1_value(key, val):
            continue
        try:
            confidence = float(fact.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        if confidence < min_conf:
            continue
        require_hitl = cfg.remember_require_hitl or confidence < 0.9
        delta = MemoryDelta(key=key, value=val, source="nudge")
        await memory.update_prompt_memory(
            ctx.request, delta, require_hitl=require_hitl
        )
        append_pending_memory_delta(ctx, delta, require_hitl=require_hitl)
        written += 1

    logger.info(
        "L1 nudge session=%s turn=%s facts=%d written=%d",
        ctx.request.session_id,
        turns,
        len(facts),
        written,
    )
    return {
        "nudged": True,
        "user_turns": turns,
        "facts_extracted": len(facts),
        "facts_written": written,
        "summarizer": type(summarizer).__name__ if summarizer else "llm",
    }
