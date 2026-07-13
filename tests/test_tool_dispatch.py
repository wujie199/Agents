import pytest

from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from core.composition.tool_dispatch import invoke_tool
from core.composition.memory_helpers import build_turn_buffer, build_checkpointer
from agent_platform.tools.adapters.tool_port_adapter import ToolPortAdapter
from agent_platform.memory.memory_tool_registration import register_memory_tools
from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)
from core.ports.skills import SkillExecutionResult


class _FakeMemory:
    async def session_search(
        self,
        query,
        context,
        limit=5,
        scope="session",
        mode=None,
        sort="newest",
        around_message_id="",
        session_link="",
    ):
        return f"hit:{query}:{context.session_id}"

    async def session_search_detail(self, query, context, limit=5, scope="session"):
        from core.ports.memory import SessionSearchResult

        return SessionSearchResult(summary=f"detail:{query}", sources=["online"])

    async def skill_search(self, query, context, limit=3):
        return []

    async def run_skill(self, skill_id, inputs, context, run_context):
        return SkillExecutionResult(
            skill_id=skill_id,
            success=True,
            outputs={"echo": inputs.get("message", "")},
            steps_executed=1,
        )

    async def resolve_entity(self, mention, context):
        from core.ports.external_memory import Entity

        if mention == "张三":
            return Entity(
                mention=mention,
                canonical_id="u001",
                display_name="张三",
            )
        return None

    async def fetch_profile_facts(self, tenant_id, user_id):
        return [{"key": "部门", "value": "研发部", "source": "ldap"}]


@pytest.mark.asyncio
async def test_invoke_tool_routes_session_search():
    tools = ToolPortAdapter()
    memory = _FakeMemory()
    register_memory_tools(tools, memory)
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
        tools=tools,
    )
    result = await invoke_tool(
        ctx, "session_search", {"query": "hello", "limit": 3}
    )
    assert result == "hit:hello:s1"

    detail = await invoke_tool(
        ctx, "session_search_detail", {"query": "hello"}
    )
    assert detail["summary"] == "detail:hello"


@pytest.mark.asyncio
async def test_invoke_tool_routes_run_skill():
    tools = ToolPortAdapter()
    memory = _FakeMemory()
    register_memory_tools(tools, memory)
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
        tools=tools,
    )
    result = await invoke_tool(
        ctx,
        "run_skill",
        {"skill_id": "example", "inputs": {"message": "hi"}},
    )
    assert result["success"] is True
    assert result["outputs"]["echo"] == "hi"


@pytest.mark.asyncio
async def test_invoke_tool_routes_l4_tools():
    tools = ToolPortAdapter()
    memory = _FakeMemory()
    register_memory_tools(tools, memory)
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
        tools=tools,
    )
    entity = await invoke_tool(ctx, "resolve_entity", {"mention": "张三"})
    assert entity["canonical_id"] == "u001"

    facts = await invoke_tool(ctx, "fetch_profile_facts", {})
    assert facts[0]["key"] == "部门"


@pytest.mark.asyncio
async def test_skill_echo_builtin():
    tools = ToolPortAdapter()
    ctx = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        trace_id="tr",
        channel="cli",
    )
    result = await tools.invoke("skill_echo", {"message": "ping"}, ctx)
    assert result == {"echo": "ping"}


@pytest.mark.asyncio
async def test_build_turn_buffer_from_config():
    class _Mem:
        async def persist_turn(self, *_a, **_k):
            pass

    assert build_turn_buffer(_Mem(), {"turn_buffer_flush_size": 0}) is None
    buf = build_turn_buffer(_Mem(), {"turn_buffer_flush_size": 5})
    assert buf is not None
    assert buf._flush_size == 5


@pytest.mark.asyncio
async def test_relational_checkpointer_save_load(tmp_path):
    db = AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_path / "cp.db"), pool_size=2
    )
    cp = build_checkpointer(db)
    assert cp is not None

    cp_id = await cp.save(
        "thread-1",
        "tenant1",
        {"node": "worker", "step": 2},
        session_id="sess-1",
    )
    assert cp_id

    loaded = await cp.load("thread-1", "tenant1")
    assert loaded is not None
    assert loaded["state"]["node"] == "worker"
    assert loaded["state"]["step"] == 2

    threads = await cp.list_threads("tenant1")
    assert threads
    assert threads[0]["thread_id"] == "thread-1"
