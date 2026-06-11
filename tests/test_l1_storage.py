"""L1 KV upsert、pending 去重、写入校验。"""

from __future__ import annotations

import pytest

from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta

from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter


def _ctx() -> RequestContext:
    return RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        trace_id="tr",
        channel="test",
    )


def test_user_kv_dedupe_upsert(tmp_path):
    hot = HotMemoryFileAdapter(store_dir=str(tmp_path / "mem"))
    hot.save_user("t1", "u1", "姓名: 张三\n职业: 工程师\n备注: 自由文本")
    hot.apply_delta("t1", "u1", MemoryDelta(key="姓名", value="李四", source="user"))
    content = hot.get_raw_user("t1", "u1")
    assert content.count("姓名:") == 1
    assert "李四" in content
    assert "职业: 工程师" in content
    assert "备注: 自由文本" in content


def test_pending_delta_dedupe_by_key(tmp_path):
    hot = HotMemoryFileAdapter(store_dir=str(tmp_path / "mem"))
    hot.queue_pending_delta(
        "t1", "u1", MemoryDelta(key="姓名", value="甲", source="user")
    )
    hot.queue_pending_delta(
        "t1", "u1", MemoryDelta(key="姓名", value="乙", source="user")
    )
    hot.queue_pending_delta(
        "t1", "u1", MemoryDelta(key="语言", value="中文", source="user")
    )
    pending = hot.list_pending_deltas("t1", "u1")
    assert len(pending) == 2
    by_key = {d.key: d.value for d in pending}
    assert by_key["姓名"] == "乙"
    assert by_key["语言"] == "中文"


@pytest.mark.asyncio
async def test_apply_memory_delta_rejects_bad_value(tmp_path):
    from datetime import datetime

    from agent_platform.infrastructure.privacy.adapter import PrivacyPortAdapter
    from agent_platform.infrastructure.skills.adapter import SimpleSkillAdapter
    from agent_platform.memory.adapters.file_external_memory_adapter import (
        FileExternalMemoryAdapter,
    )
    from agent_platform.memory.adapters.hot_memory_compressor_adapter import (
        TruncatingHotMemoryCompressorAdapter,
    )
    from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
    from agent_platform.memory.adapters.skill_memory_adapter import SkillMemoryAdapter
    from agent_platform.memory.adapters.summarizer_adapter import TruncatingSummarizerAdapter
    from agent_platform.storage.adapters.sqlite.relational_adapter import (
        AsyncSQLiteRelationalAdapter,
    )

    store = str(tmp_path / "mem")
    db = AsyncSQLiteRelationalAdapter(db_path=str(tmp_path / "a.db"), pool_size=2)
    hot = HotMemoryFileAdapter(store_dir=store)
    memory = MemoryPortAdapter(
        store_dir=store,
        archive_db=db,
        hot_memory=hot,
        privacy=PrivacyPortAdapter(),
        skill_memory=SkillMemoryAdapter(
            skills=SimpleSkillAdapter(skills_dir="skills/published"),
            drafts_dir=str(tmp_path / "drafts"),
        ),
        summarizer=TruncatingSummarizerAdapter(max_chars=500),
        compressor=TruncatingHotMemoryCompressorAdapter(),
        external_memory=FileExternalMemoryAdapter(
            profiles_dir=str(tmp_path / "ext")
        ),
    )
    ctx = _ctx()
    await memory.apply_memory_delta(
        ctx, MemoryDelta(key="输出格式", value="json", source="user")
    )
    assert hot.get_raw_user("t1", "u1") == ""

    await memory.apply_memory_delta(
        ctx, MemoryDelta(key="姓名", value="武杰", source="user")
    )
    assert "武杰" in hot.get_raw_user("t1", "u1")
