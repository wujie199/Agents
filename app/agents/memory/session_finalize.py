# -*- coding: utf-8 -*-
"""会话结束：L2→L1 抽取 + finalize。"""

from __future__ import annotations

from typing import Any, List

from core.composition.run_context import RunContext
from core.ports.memory import MemoryDelta

from app.agents.orchestration.chat_config import ChatAgentConfig, load_chat_config
from app.agents.context_builder import extract_l1_facts_from_session, validate_l1_key, is_allowed_l1_value
from app.agents.orchestration.chat_nodes import fetch_turn_history
from app.agents.debug.debug_trace import agent_debug
from app.agents.memory.conflict_detector import (
    check_l1_write_conflicts,
    ConflictStrategy,
)


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

    # ── L1 冲突检测：写入前检查现有 facts 与新 deltas 的冲突 ──
    existing_facts: dict[str, str] = {}
    try:
        snap = memory.compose_prompt_snapshot(ctx.request)
        # snapshot 的 memory_text 格式为 "key: value\n..."，解析为 dict
        for line in (snap.memory_text or "").split("\n"):
            line = line.strip()
            if ": " in line:
                k, v = line.split(": ", 1)
                existing_facts[k.strip()] = v.strip()
    except Exception:
        pass

    conflict_strategy = ConflictStrategy(cfg.l1_conflict_strategy)
    conflict_records = check_l1_write_conflicts(
        existing_facts,
        [{"key": f["key"], "value": f.get("value") or ""} for f in facts],
        strategy=conflict_strategy,
        l1_auto_write_confidence_min=cfg.l1_auto_write_confidence_min,
    )

    # 按冲突结果过滤：只写入 resolved_value == new_value 且不需 HITL 的条目
    # 需要 HITL 的写入 pending（HITL），冲突保留旧值的跳过
    conflict_map: dict[str, ConflictRecord] = {r.key: r for r in conflict_records}

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
        # 冲突过滤
        record = conflict_map.get(key)
        if record is not None:
            if record.resolved_value != val:
                # 冲突且保留了旧值，跳过写入
                agent_debug(
                    "FINALIZE-L1",
                    "session_finalize.enrich_l1_before_finalize:conflict_skip",
                    "L1 冲突保留旧值",
                    {"key": key, "old": record.old_value, "new": val, "strategy": record.strategy},
                )
                continue
            if record.needs_hitl:
                # 需 HITL 确认，写 pending
                require_hitl = True
            else:
                require_hitl = cfg.remember_require_hitl
        else:
            require_hitl = cfg.remember_require_hitl
        await memory.update_prompt_memory(
            ctx.request,
            MemoryDelta(key=key, value=val, source="user"),
            require_hitl=require_hitl,
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
