# -*- coding: utf-8 -*-
"""Phase C：L2 FTS5 + Hermes session_search 四形态测试。"""

from __future__ import annotations

import json

import pytest

from core.domain.context import RequestContext
from core.ports.memory import TurnRecord

from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter
from agent_platform.memory.adapters.l2_session_search import (
    apply_cron_downrank,
    build_bookends_narrative,
    dedupe_hits_by_lineage_async,
    parse_session_link,
    resolve_lineage_root_async,
)
from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
from agent_platform.memory.adapters.summarizer_adapter import TruncatingSummarizerAdapter
from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)


def _ctx(session: str = "current", user: str = "u1") -> RequestContext:
    return RequestContext(
        tenant_id="t1",
        user_id=user,
        session_id=session,
        trace_id="tr",
        channel="test",
    )


@pytest.fixture
def db(tmp_path):
    return AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_path / "l2.db"), pool_size=2
    )


@pytest.fixture
def memory(tmp_path, db):
    hot = HotMemoryFileAdapter(store_dir=str(tmp_path / "mem"))
    return MemoryPortAdapter(
        store_dir=str(tmp_path / "mem"),
        archive_db=db,
        hot_memory=hot,
        summarizer=TruncatingSummarizerAdapter(max_chars=500),
        session_search_rerank=False,
        session_hybrid_search=False,
        l2_compression_continuation="split",
    )


async def _seed_session(
    memory: MemoryPortAdapter,
    ctx: RequestContext,
    messages: list[tuple[str, str]],
    *,
    source: str = "user",
    title: str = "",
    parent_session_id: str | None = None,
):
    await memory.ensure_session(ctx)
    if title or parent_session_id or source != "user":
        await memory._archive_db.update(
            "sessions",
            {
                **({"title": title} if title else {}),
                **({"parent_session_id": parent_session_id} if parent_session_id else {}),
                **({"source": source} if source else {}),
            },
            {"session_id": ctx.session_id},
        )
    for i, (role, content) in enumerate(messages):
        await memory.persist_turn(
            ctx,
            TurnRecord(
                role=role,
                content=content,
                ts=f"2026-01-01T10:00:{i:02d}",
            ),
        )


@pytest.mark.asyncio
async def test_trigram_fts_discovery_finds_cjk(memory, db):
    old_ctx = _ctx(session="old_sess")
    await _seed_session(
        memory,
        old_ctx,
        [
            ("user", "扫地机器人基站无法回充"),
            ("assistant", "请检查充电触点是否脏污"),
        ],
        title="扫地机器人讨论",
    )
    current = _ctx(session="current")
    await memory.ensure_session(current)

    text = await memory.session_search(
        "扫地机器人回充",
        current,
        limit=3,
        mode="discovery",
    )
    assert "扫地机器人" in text
    assert "old_sess" in text or "@session:" in text


@pytest.mark.asyncio
async def test_discovery_skips_current_lineage(memory):
    other = _ctx(session="other_alpha")
    await _seed_session(
        memory,
        other,
        [("user", "另一个会话里的 Alpha 笔记")],
        title="Other Alpha",
    )
    parent = _ctx(session="parent")
    await _seed_session(
        memory,
        parent,
        [("user", "当前 lineage 项目 Alpha 启动")],
        title="Alpha",
    )
    await memory._archive_db.create_compression_child_session(
        "parent",
        new_session_id="child",
        tenant_id="t1",
        user_id="u1",
    )
    child = _ctx(session="child")
    await _seed_session(
        memory,
        child,
        [("user", "当前 lineage 项目 Alpha 第二阶段")],
    )

    payload = json.loads(
        await memory.session_search_hermes(child, mode="discovery", query="Alpha", limit=5)
    )
    session_ids = {r["session_id"] for r in payload.get("results", [])}
    assert "child" not in session_ids
    assert "parent" not in session_ids
    assert "other_alpha" in session_ids


@pytest.mark.asyncio
async def test_scroll_around_message(memory):
    ctx = _ctx(session="scroll_sess")
    await _seed_session(
        memory,
        ctx,
        [
            ("user", "第一条"),
            ("assistant", "第二条"),
            ("user", "第三条关键词"),
            ("assistant", "第四条"),
        ],
    )
    rows = await memory._archive_db.select_many(
        "messages",
        ["message_id", "content"],
        where={"session_id": "scroll_sess"},
        order_by="ts ASC",
    )
    anchor = next(r for r in rows if "关键词" in r["content"])
    reader = _ctx(session="other")
    await memory.ensure_session(reader)

    payload = json.loads(
        await memory.session_search_hermes(
            reader,
            mode="scroll",
            around_message_id=anchor["message_id"],
            before=1,
            after=1,
        )
    )
    assert payload["mode"] == "scroll"
    assert len(payload["messages"]) == 3
    assert any("关键词" in m["content"] for m in payload["messages"])


@pytest.mark.asyncio
async def test_read_session_link(memory):
    ctx = _ctx(session="read_sess")
    await _seed_session(
        memory,
        ctx,
        [("user", "完整会话内容 A"), ("assistant", "完整会话内容 B")],
    )
    link = "@session:read_sess"
    payload = json.loads(
        await memory.session_search_hermes(_ctx(session="x"), mode="read", session_link=link)
    )
    assert payload["mode"] == "read"
    assert len(payload["messages"]) == 2
    assert parse_session_link(link)["session_id"] == "read_sess"


@pytest.mark.asyncio
async def test_browse_lists_sessions(memory):
    for sid, title in (("s1", "会话一"), ("s2", "会话二")):
        await _seed_session(
            memory,
            _ctx(session=sid),
            [("user", f"{title} 内容")],
            title=title,
        )
    payload = json.loads(
        await memory.session_search_hermes(_ctx(session="cur"), mode="browse", limit=5)
    )
    assert payload["mode"] == "browse"
    assert len(payload["sessions"]) >= 2


@pytest.mark.asyncio
async def test_cron_source_downrank():
    hits = [
        {"session_id": "a", "score": 1.0, "session_source": "user"},
        {"session_id": "b", "score": 1.0, "source": "cron"},
    ]
    ranked = apply_cron_downrank(hits)
    assert ranked[0]["session_id"] == "a"
    assert ranked[1]["score"] < 1.0


@pytest.mark.asyncio
async def test_lineage_dedup_helpers():
    parents = {"child": "parent", "parent": None}

    async def lookup(sid: str):
        return parents.get(sid)

    root = await resolve_lineage_root_async("child", lookup)
    assert root == "parent"

    deduped = await dedupe_hits_by_lineage_async(
        [
            {"session_id": "parent", "score": 0.5, "ts": "1"},
            {"session_id": "child", "score": 0.9, "ts": "2"},
        ],
        parent_lookup=lookup,
    )
    assert len(deduped) == 1
    assert deduped[0]["session_id"] == "child"


def test_bookends_narrative():
    text = build_bookends_narrative(
        {"session_id": "s1", "title": "Demo", "started_at": "2026-01-01"},
        {"role": "user", "content": "开始", "ts": "t1"},
        {"role": "user", "content": "命中", "ts": "t2", "message_id": "m2"},
        {"role": "assistant", "content": "结束", "ts": "t3", "message_id": "m3"},
    )
    assert "[opening]" in text
    assert "[hit]" in text
    assert "[closing]" in text


@pytest.mark.asyncio
async def test_compression_split_creates_child(memory):
    parent = _ctx(session="compress_parent")
    await _seed_session(memory, parent, [("user", "before compress")], title="Proj")
    result = await memory.split_session_on_compression(parent, "compress_child")
    assert result["success"] is True
    assert result["mode"] == "split"
    assert result["parent_session_id"] == "compress_parent"
    child = await memory._archive_db.get_session("compress_child")
    assert child["parent_session_id"] == "compress_parent"
    assert "#2" in str(child.get("title") or "")


@pytest.mark.asyncio
async def test_append_message_alias(memory):
    ctx = _ctx(session="append_sess")
    await memory.append_message(
        ctx, TurnRecord(role="user", content="via append", ts="2026-01-02T10:00:00")
    )
    rows = await memory.list_turns(ctx, limit=10)
    assert len(rows) == 1
    assert rows[0]["content"] == "via append"
