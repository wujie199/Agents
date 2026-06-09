import asyncio
from pathlib import Path

import pytest

from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from agent_platform.infrastructure.skills.adapter import SimpleSkillAdapter
from agent_platform.tools.adapters.tool_port_adapter import ToolPortAdapter
from agent_platform.memory.memory_tool_registration import register_memory_tools


def _ctx(channel: str = "test") -> RequestContext:
    return RequestContext(
        tenant_id="tenant1",
        user_id="user1",
        session_id="skill_sess",
        trace_id="tr1",
        channel=channel,
    )


def _run_ctx(memory=None) -> RunContext:
    tools = ToolPortAdapter(config_path="config/tools.yml")
    if memory is not None:
        register_memory_tools(tools, memory)
    return RunContext(request=_ctx(), memory=memory, tools=tools)


@pytest.mark.asyncio
async def test_validate_tools_blocks_missing():
    adapter = SimpleSkillAdapter(skills_dir="skills/published")
    skill = adapter.get("example")
    assert skill is not None
    missing = adapter.validate_tools(skill, ["session_search"])
    assert "skill_echo" in missing

    run_ctx_full = _run_ctx()
    missing_ok = adapter.validate_tools(
        skill, run_ctx_full.tools.list_tools()
    )
    assert "skill_echo" not in missing_ok


@pytest.mark.asyncio
async def test_run_skill_timeout(tmp_path):
    slow_dir = tmp_path / "skills" / "slow"
    slow_dir.mkdir(parents=True)
    (slow_dir / "skill.yaml").write_text(
        """
skill_id: slow_skill
version: "1.0.0"
title: Slow
triggers: [slow]
required_tools: [skill_echo]
max_duration_seconds: 1
acl: [test]
steps:
  - action: wait
    tool: skill_echo
    args_template:
      message: "x"
""",
        encoding="utf-8",
    )

    tools = ToolPortAdapter()

    async def slow_echo(message: str = "", **kwargs):
        await asyncio.sleep(2)
        return {"echo": message}

    tools.register_tool("skill_echo", slow_echo, acl=["user", "test"])
    adapter = SimpleSkillAdapter(skills_dir=str(tmp_path / "skills"))
    run_ctx = RunContext(request=_ctx(), tools=tools)

    result = await adapter.run("slow_skill", {}, run_ctx)
    assert not result.success
    assert "timed out" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_json_section_skill(tmp_path):
    read_json = tmp_path / "read.json"
    read_json.write_text(
        '{"sections": {"intro": {"text": "hello"}}}',
        encoding="utf-8",
    )
    save_json = tmp_path / "out.json"

    skill_dir = tmp_path / "skills" / "custom_json"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.yaml").write_text(
        f"""
skill_id: custom_json
version: "1.0.0"
title: Custom JSON
triggers: [custom]
required_tools: [read_json_context_by_title, save_result_2_json]
max_duration_seconds: 30
acl: [test]
steps:
  - action: read
    tool: read_json_context_by_title
    args_template:
      title: "intro"
      json_path: "{read_json}"
  - action: save
    tool: save_result_2_json
    args_template:
      payload: "$outputs.read"
      json_path: "{save_json}"
""",
        encoding="utf-8",
    )

    adapter = SimpleSkillAdapter(skills_dir=str(tmp_path / "skills"))
    run_ctx = _run_ctx()
    result = await adapter.run("custom_json", {}, run_ctx)
    assert result.success, result.error
    assert save_json.exists()
    saved = save_json.read_text(encoding="utf-8")
    assert "intro" in saved


@pytest.mark.asyncio
async def test_skill_step_invokes_memory_tool(tmp_path):
    class _FakeMemory:
        async def skill_search(self, query, context, limit=3):
            from core.ports.memory import SkillSummary

            return [
                SkillSummary(
                    skill_id="example",
                    title="Example",
                    summary="demo",
                    success_rate=1.0,
                )
            ]

    probe_dir = tmp_path / "skills" / "probe_memory"
    probe_dir.mkdir(parents=True)
    (probe_dir / "skill.yaml").write_text(
        """
skill_id: probe_memory
version: "1.0.0"
title: Probe
triggers: [probe]
required_tools: [skill_search]
max_duration_seconds: 30
acl: [test]
steps:
  - action: search
    tool: skill_search
    args_template:
      query: "报告"
      limit: 3
""",
        encoding="utf-8",
    )

    adapter = SimpleSkillAdapter(skills_dir=str(tmp_path / "skills"))
    run_ctx = _run_ctx(memory=_FakeMemory())
    result = await adapter.run("probe_memory", {}, run_ctx)
    assert result.success, result.error
    assert len(result.outputs["search"]) == 1
