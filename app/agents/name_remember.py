# -*- coding: utf-8 -*-
"""姓名/称呼自述：确定性写入 L1 pending（HITL）。"""

from __future__ import annotations

import re
from typing import Optional

from core.composition.run_context import RunContext
from core.ports.memory import MemoryDelta

from app.agents.chat_config import ChatAgentConfig
from app.agents.context_builder import (
    is_allowed_l1_value,
    is_name_intro_query,
    validate_l1_key,
)

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
    await memory.update_prompt_memory(
        ctx.request,
        MemoryDelta(key=key, value=name, source="user"),
        require_hitl=cfg.remember_require_hitl,
    )
    if cfg.remember_require_hitl:
        from app.agents.memory_metrics import record_l1_pending

        record_l1_pending(ctx)
    return f"{key}={name}"
