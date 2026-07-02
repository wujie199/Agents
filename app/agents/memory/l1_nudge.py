# -*- coding: utf-8 -*-
"""L1 后台记忆回顾 nudge（Phase B stub）。"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.composition.run_context import RunContext

logger = logging.getLogger(__name__)

_NUDGE_TURN_KEY = "_l1_nudge_user_turns"


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
) -> Optional[dict[str, Any]]:
    """每 nudge_interval 个 user turn 评估是否写入记忆；无 summarizer 时跳过。"""
    if nudge_interval <= 0:
        return None
    turns = record_user_turn_for_nudge(ctx)
    if turns % nudge_interval != 0:
        return None
    if summarizer is None:
        logger.debug(
            "L1 nudge skipped (no summarizer) session=%s turn=%s",
            ctx.request.session_id,
            turns,
        )
        return {"skipped": True, "reason": "no_summarizer", "user_turns": turns}
    logger.debug(
        "L1 nudge hook fired session=%s turn=%s (summarizer stub)",
        ctx.request.session_id,
        turns,
    )
    return {"nudged": True, "user_turns": turns, "action": "stub"}
