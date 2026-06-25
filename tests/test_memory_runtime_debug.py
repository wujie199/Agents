"""记忆运行状态采集单元测试。"""

from __future__ import annotations

import pytest

from core.composition.run_context import RunContext
from core.domain.context import RequestContext

from app.agents.memory.memory_runtime_debug import (
    collect_memory_runtime_status,
    set_memory_runtime_debug,
)


@pytest.mark.asyncio
async def test_collect_memory_runtime_status_minimal():
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
    )
    data = await collect_memory_runtime_status(ctx, event="test")
    assert data["event"] == "test"
    assert "layers" in data
    assert "error" in data["layers"] or data["layers"].get("L1")


@pytest.mark.asyncio
async def test_collect_memory_runtime_status_verbose(tmp_path, monkeypatch):
    from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter

    store = tmp_path / "mem"
    hot = HotMemoryFileAdapter(store_dir=str(store), use_file_lock=False)
    user_path = store / "t1" / "USER_u1.md"
    user_path.parent.mkdir(parents=True, exist_ok=True)
    user_path.write_text("称呼: 测试\n", encoding="utf-8")

    class _Mem:
        _hot = hot

        def compose_prompt_snapshot(self, req):
            return hot.compose_snapshot(req)

        async def list_turns(self, req, limit=10):
            return [{"role": "user", "content": "hi", "ts": "now"}]

    from app.agents.memory.memory_runtime_debug import (
        collect_memory_runtime_status,
        set_memory_runtime_verbose,
    )

    set_memory_runtime_verbose(True)
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        memory=_Mem(),
    )
    data = await collect_memory_runtime_status(ctx, event="verbose_test", verbose=True)
    l1 = data["layers"]["L1"]
    assert l1.get("memory_preview")
    assert l1.get("files", {}).get("user_md", {}).get("exists") is True
    assert data.get("verbose") is True
    set_memory_runtime_verbose(False)


    set_memory_runtime_verbose(False)


def test_resolve_memory_trace_dev_off_by_default():
    from app.agents.memory.memory_runtime_debug import resolve_memory_trace

    on, console = resolve_memory_trace(profile="dev")
    assert on is False
    assert console is False


def test_resolve_memory_trace_debug_flag():
    from app.agents.memory.memory_runtime_debug import resolve_memory_trace

    on, console = resolve_memory_trace(profile="dev", debug=True)
    assert on is True
    assert console is True


def test_resolve_memory_trace_no_debug():
    from app.agents.memory.memory_runtime_debug import resolve_memory_trace

    on, console = resolve_memory_trace(profile="dev", no_debug=True)
    assert on is False
    assert console is False


def test_format_detailed_l4_dict_facts():
    from app.agents.memory.memory_runtime_debug import format_memory_runtime_detailed

    data = {
        "event": "test",
        "request": {"session_id": "s1", "tenant_id": "t1", "user_id": "u1"},
        "layers": {
            "L1": {"hash": "abc", "chars": 10, "backend": "file"},
            "L2": {"archive_backend": "sqlite", "session_turn_rows": 2},
            "L3": {"published_count": 1},
            "L4": {
                "backend": "file",
                "facts_count": 2,
                "facts": [
                    {"key": "部门", "value": "研发部", "source": "ldap"},
                    {"key": "职位", "value": "工程师", "source": "hr"},
                ],
            },
        },
        "infrastructure": {},
        "rag": {"port_present": True},
    }
    text = format_memory_runtime_detailed(data)
    assert "部门" in text
    assert "研发部" in text
    assert "L4 外部画像" in text


@pytest.mark.asyncio
async def test_log_writes_when_enabled(tmp_path, monkeypatch):
    log_file = tmp_path / "mem-debug.log"
    monkeypatch.setenv("MEMORY_DEBUG_LOG", str(log_file))
    set_memory_runtime_debug(True)
    from app.agents.memory.memory_runtime_debug import log_memory_runtime_status

    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
    )
    await log_memory_runtime_status(ctx, event="unit_test")
    assert log_file.is_file()
    assert "MEM-RUNTIME" in log_file.read_text(encoding="utf-8")
    set_memory_runtime_debug(False)


@pytest.mark.asyncio
async def test_log_force_without_enabled(tmp_path, monkeypatch):
    log_file = tmp_path / "mem-force.log"
    monkeypatch.setenv("MEMORY_DEBUG_LOG", str(log_file))
    set_memory_runtime_debug(False)
    from app.agents.memory.memory_runtime_debug import log_memory_runtime_status

    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
    )
    await log_memory_runtime_status(ctx, event="forced", force=True)
    assert log_file.is_file()
    assert "forced" in log_file.read_text(encoding="utf-8")
