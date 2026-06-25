import pytest

from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)
from agent_platform.memory.adapters.relational_checkpointer_adapter import (
    RelationalCheckpointerAdapter,
)
from app.agents.roles.react_loop import end_agent_session


@pytest.mark.asyncio
async def test_end_agent_session_saves_checkpoint(tmp_path):
    db = AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_path / "archive.db"), pool_size=2
    )
    checkpointer = RelationalCheckpointerAdapter(db)

    class _FakeMemory:
        async def end_session(self, request, status="closed", finalize=True):
            self.last_status = status

        async def list_turns(self, request, limit=50):
            return []

        def get_snapshot_hash(self, tenant_id, user_id):
            return "snap-hash"

    memory = _FakeMemory()
    ctx = RunContext(
        request=RequestContext(
            tenant_id="tenant1",
            user_id="user1",
            session_id="sess-cp",
            trace_id="t1",
            channel="test",
        ),
        memory=memory,
        checkpointer=checkpointer,
    )

    await end_agent_session(
        ctx,
        checkpoint_state={"done": True, "node": "finalize"},
    )

    loaded = await checkpointer.load("sess-cp", "tenant1")
    assert loaded["state"]["done"] is True
    assert loaded["state"]["memory_snapshot_hash"] == "snap-hash"
    assert memory.last_status == "closed"


@pytest.mark.asyncio
async def test_end_agent_session_default_checkpoint_state(tmp_path):
    db = AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_path / "archive.db"), pool_size=2
    )
    checkpointer = RelationalCheckpointerAdapter(db)

    class _FakeMemory:
        async def end_session(self, request, status="closed", finalize=True):
            pass

        async def list_turns(self, request, limit=50):
            return []

        def get_snapshot_hash(self, tenant_id, user_id):
            return "auto-hash"

    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="sess-auto",
            trace_id="tr",
            channel="test",
        ),
        memory=_FakeMemory(),
        checkpointer=checkpointer,
    )
    from app.agents.orchestration.chat_config import ChatAgentConfig

    await end_agent_session(
        ctx,
        chat_cfg=ChatAgentConfig(enable_l1_extract_on_finalize=False),
    )
    loaded = await checkpointer.load("sess-auto", "t1")
    assert loaded["state"]["memory_snapshot_hash"] == "auto-hash"
    assert loaded["state"]["session_id"] == "sess-auto"
