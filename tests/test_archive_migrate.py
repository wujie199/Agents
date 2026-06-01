import pytest
from datetime import datetime

from core.ports.memory import TurnRecord
from core.domain.context import RequestContext
from agent_platform.memory.adapters.archive_migrate import migrate_sqlite_to_postgresql
from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter
from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)


def _ctx(session_id: str = "mig1") -> RequestContext:
    return RequestContext(
        tenant_id="tenant1",
        user_id="user1",
        session_id=session_id,
        trace_id="trace1",
        channel="test",
    )


@pytest.mark.asyncio
async def test_migrate_sqlite_dry_run(tmp_path):
    src_path = tmp_path / "source.db"
    dst_path = tmp_path / "dest.db"
    src = AsyncSQLiteRelationalAdapter(db_path=str(src_path), pool_size=2)
    dst = AsyncSQLiteRelationalAdapter(db_path=str(dst_path), pool_size=2)

    memory = MemoryPortAdapter(archive_db=src, hot_memory=HotMemoryFileAdapter(store_dir=str(tmp_path / "mem")))
    ctx = _ctx()
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(role="user", content="migrate me", ts=datetime.now().isoformat()),
    )

    stats = await migrate_sqlite_to_postgresql(src, dst, dry_run=True)
    assert stats["sessions"] == 1
    assert stats["messages"] == 1
    assert stats["sessions_written"] == 0

    rows = await dst.select_many("sessions", ["session_id"], limit=10)
    assert len(rows) == 0

    await src.close()
    await dst.close()


@pytest.mark.asyncio
async def test_migrate_sqlite_to_sqlite_copy(tmp_path):
    """无 PG 时用第二个 SQLite 验证写入逻辑。"""
    src_path = tmp_path / "source.db"
    dst_path = tmp_path / "dest.db"
    src = AsyncSQLiteRelationalAdapter(db_path=str(src_path), pool_size=2)
    dst = AsyncSQLiteRelationalAdapter(db_path=str(dst_path), pool_size=2)

    memory = MemoryPortAdapter(
        archive_db=src, hot_memory=HotMemoryFileAdapter(store_dir=str(tmp_path / "mem"))
    )
    ctx = _ctx("copy_sess")
    await memory.persist_turn(
        ctx,
        TurnRecord(role="assistant", content="hello pg", ts=datetime.now().isoformat()),
    )

    stats = await migrate_sqlite_to_postgresql(src, dst, dry_run=False)
    assert stats["messages_written"] == 1
    assert stats["errors"] == 0

    row = await dst.select_one(
        "messages", ["content"], {"session_id": "copy_sess"}
    )
    assert row["content"] == "hello pg"

    await src.close()
    await dst.close()
