# -*- coding: utf-8 -*-
"""记忆只读视图（CLI / API 辅助）。"""

from __future__ import annotations

from typing import Any, Dict, List

from core.composition.run_context import RunContext


def list_pending_l1_deltas(ctx: RunContext) -> List[Dict[str, str]]:
    """列出待确认 L1 pending（HITL）。"""
    memory = ctx.memory
    if memory is None:
        return []
    hot = getattr(memory, "_hot", None)
    if hot is None or not hasattr(hot, "list_pending_deltas"):
        return []
    req = ctx.request
    raw = hot.list_pending_deltas(req.tenant_id, req.user_id)
    out: List[Dict[str, str]] = []
    for item in raw or []:
        if hasattr(item, "key"):
            out.append(
                {
                    "key": str(getattr(item, "key", "")),
                    "value": str(getattr(item, "value", "")),
                    "source": str(getattr(item, "source", "")),
                }
            )
        elif isinstance(item, dict):
            out.append(
                {
                    "key": str(item.get("key", "")),
                    "value": str(item.get("value", "")),
                    "source": str(item.get("source", "")),
                }
            )
    return out
