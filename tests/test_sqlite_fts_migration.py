# -*- coding: utf-8 -*-
"""Legacy FTS schema migration (content-only → message_id + session_id)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import pytest

from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)


@pytest.mark.asyncio
async def test_rebuild_legacy_trigram_fts_schema(tmp_path: Path) -> None:
    db_path = str(tmp_path / "legacy.db")
    async with aiosqlite.connect(db_path) as conn:
        await conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                started_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                ts TEXT NOT NULL,
                redacted INTEGER DEFAULT 0
            );
            INSERT INTO sessions VALUES ('s1', 'u1', 't1', '2026-01-01');
            INSERT INTO messages VALUES ('m1', 's1', 'user', 'hello world', '2026-01-01', 0);
            CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
                content,
                tokenize='unicode61'
            );
            INSERT INTO messages_fts_trigram (content) VALUES ('hello world');
            """
        )
        await conn.commit()

    adapter = AsyncSQLiteRelationalAdapter(db_path=db_path, pool_size=1)
    await adapter._init_pool()

    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute("PRAGMA table_info(messages_fts_trigram)")
        cols = [row[1] for row in await cursor.fetchall()]
    assert "message_id" in cols
    assert "session_id" in cols

    rows = await adapter.search_messages_ranked(tenant_id="t1", query="hello", limit=5)
    assert len(rows) >= 1
    assert rows[0]["message_id"] == "m1"
