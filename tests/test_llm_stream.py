"""LLM 流式适配单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.prompts.llm_stream import extract_stream_delta, stream_llm_text


def test_extract_stream_delta_openai_chunk():
    delta = MagicMock()
    delta.content = "你好"
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    assert extract_stream_delta(chunk) == "你好"


def test_extract_stream_delta_string():
    assert extract_stream_delta("片段") == "片段"


@pytest.mark.asyncio
async def test_stream_llm_text_uses_astream():
    llm = MagicMock()

    async def _astream(_messages):
        yield "a"
        yield "b"

    llm.astream = _astream
    parts = [p async for p in stream_llm_text(llm, [{"role": "user", "content": "hi"}])]
    assert parts == ["a", "b"]


@pytest.mark.asyncio
async def test_stream_llm_text_fallback_ainvoke():
    llm = MagicMock()
    llm.astream = None
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "完整回复"
    llm.ainvoke = AsyncMock(return_value=resp)
    parts = [p async for p in stream_llm_text(llm, [{"role": "user", "content": "hi"}])]
    assert parts == ["完整回复"]
