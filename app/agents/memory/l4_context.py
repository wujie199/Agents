# -*- coding: utf-8 -*-
"""L4 MemoryManager prefetch 注入（瞬态 user payload）。"""

from __future__ import annotations

from typing import Any, List, Optional

from core.composition.run_context import RunContext
from core.ports.observability import Layer

from agent_platform.memory.adapters.memory_manager import MemoryManager
from app.agents.observability.instrument import span_ctx


def get_memory_manager(ctx: RunContext) -> Optional[MemoryManager]:
    extra = getattr(ctx, "extra", None)
    if not isinstance(extra, dict):
        return None
    mgr = extra.get("memory_manager")
    return mgr if isinstance(mgr, MemoryManager) else None


async def apply_l4_prefetch_to_messages(
    ctx: RunContext,
    messages: List[dict[str, Any]],
    user_message: str,
) -> List[dict[str, Any]]:
    span_attrs: dict[str, Any] = {}
    async with span_ctx(ctx, "memory.l4.prefetch", Layer.AGENT, span_attrs):
        mgr = get_memory_manager(ctx)
        if mgr is None:
            return messages
        snippets = await mgr.prefetch_all(user_message, ctx.request)
        span_attrs["snippet_count"] = len(snippets or [])
        if not snippets:
            return messages
        block = mgr.build_memory_context_block(snippets, ctx.request.tenant_id)
        if not block:
            return messages
        span_attrs["injected"] = True
        return MemoryManager.inject_memory_context_into_messages(messages, block)
