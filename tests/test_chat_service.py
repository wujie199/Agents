"""chat_service 单元测试。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.domain.context import RequestContext
from core.composition.run_context import RunContext

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.orchestration.chat_service import (
    ChatSessionHandle,
    execute_chat_turn,
    stream_chat_turn_events,
)

from app.agents.orchestration.chat_turn import ChatTurnResult


def _handle() -> ChatSessionHandle:
    return ChatSessionHandle(
        run_ctx=RunContext(
            request=RequestContext(
                tenant_id="t",
                user_id="u",
                session_id="s",
                trace_id="tr",
                channel="test",
            )
        ),
        chat_cfg=ChatAgentConfig(enable_memory_tools=False),
    )


@pytest.mark.asyncio
async def test_execute_chat_turn_direct():
    handle = _handle()
    mock_result = ChatTurnResult(assistant_text="ok", evidence_count=0)
    with patch(
        "app.agents.orchestration.chat_service.run_chat_turn",
        new=AsyncMock(return_value=mock_result),
    ) as mock_run:
        result = await execute_chat_turn(handle, "hi", engine="direct")
    assert result.assistant_text == "ok"
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_chat_turn_events():
    handle = _handle()
    mock_result = ChatTurnResult(
        assistant_text="hello world",
        evidence_count=2,
        rag_empty=False,
        history_turns=3,
    )
    with patch(
        "app.agents.orchestration.chat_service.execute_chat_turn",
        new=AsyncMock(return_value=mock_result),
    ):
        events = [
            json.loads(x)
            async for x in stream_chat_turn_events(
                handle, "hi", engine="langgraph", stream_mode="batch"
            )
        ]
    assert events[0]["type"] == "meta"
    assert events[0]["evidence_count"] == 2
    assert any(e["type"] == "delta" for e in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["assistant_text"] == "hello world"


@pytest.mark.asyncio
async def test_stream_token_mode_direct():
    handle = _handle()
    with patch(
        "app.agents.orchestration.chat_service.build_turn_messages",
        new=AsyncMock(
            return_value=([{"role": "user", "content": "hi"}], 1, False, "abc123")
        ),
    ), patch(
        "app.agents.orchestration.chat_service.stream_llm_text",
    ) as mock_stream, patch(
        "app.agents.orchestration.chat_service.persist_user_and_assistant",
        new=AsyncMock(),
    ), patch(
        "app.agents.orchestration.chat_nodes.fetch_turn_history",
        new=AsyncMock(return_value=[]),
    ):
        async def _gen(*_a, **_k):
            yield "你"
            yield "好"

        mock_stream.return_value = _gen()

        memory = MagicMock()
        handle.run_ctx.memory = memory
        handle.run_ctx.models = MagicMock()
        handle.run_ctx.models.get_model = MagicMock(return_value=MagicMock())

        events = [
            json.loads(x)
            async for x in stream_chat_turn_events(
                handle, "hi", engine="direct", stream_mode="token"
            )
        ]
    assert events[0]["stream"] == "token"
    assert any(e.get("text") == "你" for e in events if e["type"] == "delta")
    assert events[-1]["assistant_text"] == "你好"
