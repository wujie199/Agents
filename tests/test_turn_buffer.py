import pytest

from core.domain.context import RequestContext
from core.ports.memory import TurnRecord
from agent_platform.memory.adapters.turn_buffer import TurnBuffer


class _FakeMemory:
    def __init__(self):
        self.turns = []

    async def persist_turn(self, context, turn):
        self.turns.append((context.session_id, turn.role, turn.content))


@pytest.mark.asyncio
async def test_turn_buffer_flushes_on_size():
    mem = _FakeMemory()
    buf = TurnBuffer(mem, flush_size=2)
    ctx = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        trace_id="tr",
        channel="test",
    )
    await buf.append(ctx, TurnRecord(role="user", content="a"))
    assert len(mem.turns) == 0
    await buf.append(ctx, TurnRecord(role="assistant", content="b"))
    assert len(mem.turns) == 2


@pytest.mark.asyncio
async def test_turn_buffer_pending_turns_for_session():
    mem = _FakeMemory()
    buf = TurnBuffer(mem, flush_size=10)
    ctx = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        trace_id="tr",
        channel="test",
    )
    other = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s2",
        trace_id="tr",
        channel="test",
    )
    await buf.append(ctx, TurnRecord(role="user", content="hello"))
    await buf.append(other, TurnRecord(role="user", content="other"))
    pending = buf.pending_turns_for(ctx)
    assert len(pending) == 1
    assert pending[0]["role"] == "user"
    assert pending[0]["content"] == "hello"
