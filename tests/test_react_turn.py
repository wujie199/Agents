"""react_turn 与 direct 引擎 Path B 单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from core.composition.factory import FakeModelPort
from core.domain.evidence import Evidence, EvidenceBundle, SourceType
from core.ports.memory import PromptMemorySnapshot

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.orchestration.chat_turn import run_chat_turn
from app.agents.roles.react_turn import dict_messages_to_lc, last_ai_text


def test_dict_messages_to_lc_roles():
    msgs = dict_messages_to_lc(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
    )
    assert len(msgs) == 3
    assert msgs[0].type == "system"
    assert msgs[-1].type == "ai"


@pytest.mark.asyncio
async def test_run_chat_turn_uses_react_when_tools_enabled():
    request = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s_react",
        trace_id="tr",
        channel="test",
    )
    memory = MagicMock()
    memory.compose_prompt_snapshot.return_value = PromptMemorySnapshot(
        memory_text="L1", hash="h"
    )
    memory.ensure_session = AsyncMock()
    memory.persist_turn = AsyncMock()
    memory.list_turns = AsyncMock(return_value=[])

    bundle = EvidenceBundle(
        evidences=[
            Evidence(
                id="1",
                content="ev",
                source_type=SourceType.VECTOR,
                score=0.9,
            )
        ],
        empty=False,
    )
    rag = MagicMock()
    rag.route_and_retrieve = AsyncMock(return_value=bundle)

    ctx = RunContext(
        request=request,
        memory=memory,
        rag=rag,
        models=FakeModelPort({"main_llm": MagicMock()}),
    )
    cfg = ChatAgentConfig(
        enable_rag=True,
        enable_memory_tools=True,
        enable_rag_gating=False,
    )

    with patch(
        "app.agents.orchestration.chat_turn.invoke_react_agent",
        new=AsyncMock(return_value="ReAct 回复"),
    ) as mock_react:
        result = await run_chat_turn(ctx, "问题", chat_cfg=cfg)

    assert result.assistant_text == "ReAct 回复"
    mock_react.assert_awaited_once()
