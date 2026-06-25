# -*- coding: utf-8 -*-
"""姓名/称呼自述：确定性写入 L1 pending（HITL）。"""

from __future__ import annotations

import re
from typing import Optional

from core.composition.run_context import RunContext
from core.ports.memory import MemoryDelta

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.context_builder import (
    is_allowed_l1_value,
    is_name_intro_query,
    validate_l1_key,
)
from app.agents.memory.conflict_detector import resolve_conflict, ConflictStrategy

_NAME_PATTERNS = (
    re.compile(r"^我叫(.+)$"),
    re.compile(r"^我的名字是(.+)$"),
    re.compile(r"^名字是(.+)$"),
    re.compile(r"^叫([\u4e00-\u9fa5a-zA-Z]{1,12})$"),
    re.compile(r"^我是([\u4e00-\u9fa5]{1,2})$"),
)


def parse_name_from_intro(query: str) -> Optional[str]:
    """从自述句解析姓名；无法解析时返回 None。"""
    q = (query or "").strip().rstrip("。！!？?~")
    if not q:
        return None
    for pat in _NAME_PATTERNS:
        m = pat.match(q)
        if not m:
            continue
        name = (m.group(1) or "").strip()
        if not name or len(name) > 20:
            continue
        if name in {"谁", "什么", "哪个", "哪位"}:
            continue
        return name
    return None


async def auto_remember_name_intro(
    ctx: RunContext,
    query: str,
    cfg: ChatAgentConfig,
    *,
    intent: str = "",
) -> Optional[str]:
    """
    profile/姓名自述时自动 remember_user_fact → pending L1。
    返回 pending 条目描述（key=value），未写入时返回 None。
    """
    if not cfg.enable_memory_tools:
        return None
    if intent and intent != "profile":
        return None
    if not is_name_intro_query(query):
        return None
    name = parse_name_from_intro(query)
    if not name:
        return None
    key = validate_l1_key("姓名", cfg.l1_extract_allowed_keys)
    if not is_allowed_l1_value(key, name):
        return None
    memory = ctx.memory
    if memory is None:
        return None

    # ── 冲突检测：现有 L1 中已有不同姓名时按策略处理 ──
    try:
        snap = memory.compose_prompt_snapshot(ctx.request)
        existing_facts: dict[str, str] = {}
        for line in (snap.memory_text or "").split("\n"):
            line = line.strip()
            if ": " in line:
                k, v = line.split(": ", 1)
                existing_facts[k.strip()] = v.strip()
        old_name = existing_facts.get("姓名", "")
        if old_name:
            conflict = resolve_conflict(
                "姓名", old_name, name,
                strategy=ConflictStrategy(cfg.l1_conflict_strategy),
                l1_auto_write_confidence_min=cfg.l1_auto_write_confidence_min,
            )
            if conflict.resolved_value != name:
                # 保留旧值，不写入
                return None
    except Exception:
        pass

    await memory.update_prompt_memory(
        ctx.request,
        MemoryDelta(key=key, value=name, source="user"),
        require_hitl=cfg.remember_require_hitl,
    )
    if cfg.remember_require_hitl:
        from app.agents.memory.memory_metrics import record_l1_pending

        record_l1_pending(ctx)
    return f"{key}={name}"
