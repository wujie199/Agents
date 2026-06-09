"""resolve_chat_checkpointer 单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.domain.context import RequestContext
from core.composition.run_context import RunContext

from app.runtime.adapters.langgraph.checkpointer import (
    resolve_chat_checkpointer,
    resolve_chat_checkpointer_async,
    teardown_postgres_checkpointer,
)


class _FakeRelational:
    def __init__(self, db_path: str):
        self._db_path = db_path


def test_resolve_checkpointer_from_relational(tmp_path):
    db = tmp_path / "archive.db"
    db.touch()
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t",
            user_id="u",
            session_id="s",
            trace_id="tr",
            channel="test",
        ),
        extra={"relational": _FakeRelational(str(db))},
    )
    cp = resolve_chat_checkpointer(ctx)
    assert cp is not None
    # sqlite 包未安装时回退 MemorySaver
    assert hasattr(cp, "aget_tuple") or hasattr(cp, "get_tuple")


def test_resolve_checkpointer_explicit_path(tmp_path):
    cp_path = tmp_path / "custom_cp.db"
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t",
            user_id="u",
            session_id="s",
            trace_id="tr",
            channel="test",
        ),
        extra={"langgraph_checkpoint_path": str(cp_path)},
    )
    cp = resolve_chat_checkpointer(ctx)
    assert cp is not None
    # sqlite 包未安装时回退 MemorySaver
    assert hasattr(cp, "aget_tuple") or hasattr(cp, "get_tuple")


@pytest.mark.asyncio
async def test_resolve_async_falls_back_without_pg():
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t",
            user_id="u",
            session_id="s",
            trace_id="tr",
            channel="test",
        ),
        extra={"memory_config_summary": {"archive_backend": "sqlite"}},
    )
    cp = await resolve_chat_checkpointer_async(ctx)
    assert cp is not None
    await teardown_postgres_checkpointer()


@pytest.mark.asyncio
async def test_resolve_async_uses_pg_when_configured():
    fake_saver = object()
    with patch(
        "app.runtime.adapters.langgraph.checkpointer.setup_postgres_checkpointer",
        new=AsyncMock(return_value=fake_saver),
    ):
        ctx = RunContext(
            request=RequestContext(
                tenant_id="t",
                user_id="u",
                session_id="s",
                trace_id="tr",
                channel="test",
            ),
            extra={"memory_config_summary": {"archive_backend": "postgresql"}},
        )
        cp = await resolve_chat_checkpointer_async(ctx)
    assert cp is fake_saver
    await teardown_postgres_checkpointer()
