"""memory_views 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

from core.domain.context import RequestContext
from core.composition.run_context import RunContext

from app.agents.memory.memory_views import list_pending_l1_deltas


def _ctx(memory: MagicMock) -> RunContext:
    return RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
    )


def test_list_pending_empty_when_no_hot():
    assert list_pending_l1_deltas(_ctx(MagicMock())) == []


def test_list_pending_reads_hot_adapter():
    memory = MagicMock()
    hot = MagicMock()
    hot.list_pending_deltas.return_value = [
        {"key": "称呼", "value": "老王", "source": "user"},
    ]
    memory._hot = hot
    rows = list_pending_l1_deltas(_ctx(memory))
    assert rows == [{"key": "称呼", "value": "老王", "source": "user"}]
    hot.list_pending_deltas.assert_called_once_with("t1", "u1")
