# -*- coding: utf-8 -*-
"""Web L1 斜杠命令测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.composition.run_context import RunContext
from core.domain.context import RequestContext
from core.ports.memory import PromptMemorySnapshot

from app.web.memory_commands import (
    is_memory_slash_command,
    try_handle_memory_slash_command,
)


def _ctx() -> RunContext:
    return RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="web",
        ),
        memory=MagicMock(),
    )


@pytest.mark.parametrize(
    "message,expected",
    [
        ("/confirm", True),
        ("/pending", True),
        ("/Confirm", True),
        ("/confirm extra", True),
        ("/status", False),
        ("hello", False),
    ],
)
def test_is_memory_slash_command(message, expected):
    assert is_memory_slash_command(message) is expected


@pytest.mark.asyncio
async def test_pending_empty():
    ctx = _ctx()
    with patch(
        "app.web.memory_commands.list_pending_l1_deltas", return_value=[]
    ):
        out = await try_handle_memory_slash_command(ctx, "/pending")
    assert out == "（当前无待确认的 L1 记忆）"


@pytest.mark.asyncio
async def test_pending_lists_rows():
    ctx = _ctx()
    pending = [{"key": "姓名", "value": "武杰", "source": "user"}]
    with patch(
        "app.web.memory_commands.list_pending_l1_deltas", return_value=pending
    ):
        out = await try_handle_memory_slash_command(ctx, "/pending")
    assert "武杰" in out
    assert "/confirm" in out


@pytest.mark.asyncio
async def test_confirm_applies_and_refreshes_snapshot():
    ctx = _ctx()
    pending = [{"key": "姓名", "value": "武杰", "source": "user"}]
    snap = PromptMemorySnapshot(memory_text="姓名: 武杰", hash="abc123", frozen=True)
    ctx.memory.compose_prompt_snapshot.return_value = snap

    with patch(
        "app.web.memory_commands.list_pending_l1_deltas", return_value=pending
    ), patch(
        "app.web.memory_commands.confirm_pending_l1", new_callable=AsyncMock, return_value=1
    ) as mock_confirm:
        out = await try_handle_memory_slash_command(ctx, "/confirm")

    mock_confirm.assert_awaited_once_with(ctx)
    assert "已确认 1 条" in out
    assert "abc123" in out
    ctx.memory.compose_prompt_snapshot.assert_called_once_with(ctx.request)


@pytest.mark.asyncio
async def test_non_command_returns_none():
    ctx = _ctx()
    out = await try_handle_memory_slash_command(ctx, "我叫武杰")
    assert out is None
