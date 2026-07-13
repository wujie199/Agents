# -*- coding: utf-8 -*-
"""Web 聊天：L1 记忆斜杠命令（与 REPL /confirm、/pending 对齐）。"""

from __future__ import annotations

from typing import Optional

from core.composition.run_context import RunContext

from app.agents.memory.enterprise_memory import confirm_pending_l1
from app.agents.memory.memory_views import list_pending_l1_deltas

_MEMORY_SLASH_COMMANDS = frozenset({"/confirm", "/pending"})


def is_memory_slash_command(message: str) -> bool:
    line = (message or "").strip()
    if not line.startswith("/"):
        return False
    cmd = line.split(maxsplit=1)[0].lower()
    return cmd in _MEMORY_SLASH_COMMANDS


def _format_pending_list(pending: list[dict]) -> str:
    lines = ["**待确认 L1 记忆**", ""]
    for row in pending:
        key = row.get("key") or ""
        value = row.get("value") or ""
        source = row.get("source") or ""
        suffix = f"  `[{source}]`" if source else ""
        lines.append(f"- `{key}` = {value}{suffix}")
    lines.append("")
    lines.append("输入 `/confirm` 确认写入 L1，或点击「结束当前会话」在 finalize 时一并落盘。")
    return "\n".join(lines)


async def try_handle_memory_slash_command(
    ctx: RunContext, message: str
) -> Optional[str]:
    """处理 /confirm、/pending；非记忆斜杠命令返回 None。"""
    line = (message or "").strip()
    if not is_memory_slash_command(line):
        return None

    cmd = line.split(maxsplit=1)[0].lower()
    pending = list_pending_l1_deltas(ctx)

    if cmd == "/pending":
        if not pending:
            return "（当前无待确认的 L1 记忆）"
        return _format_pending_list(pending)

    if cmd == "/confirm":
        if not pending:
            return "（当前无待确认的 L1 记忆，无需确认）"
        preview = _format_pending_list(pending)
        n = await confirm_pending_l1(ctx)
        memory = ctx.require_memory()
        snap = memory.compose_prompt_snapshot(ctx.request)
        return (
            f"{preview}\n\n"
            f"✅ **已确认 {n} 条**，已写入 L1（hash=`{snap.hash}`）。"
            f"下一轮对话将使用更新后的用户画像。"
        )

    return None
