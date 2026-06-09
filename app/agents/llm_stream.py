# -*- coding: utf-8 -*-
"""LLM 流式输出适配。"""

from __future__ import annotations

from typing import Any, AsyncIterator, List

from app.agents.react_loop import _extract_llm_text


def extract_stream_delta(chunk: Any) -> str:
    """从 astream chunk 提取文本增量。"""
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "choices") and chunk.choices:
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None)
            if content:
                return str(content)
        msg = getattr(choice, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if content:
                return str(content)
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                return str(delta["content"])
            msg = choices[0].get("message") or {}
            if msg.get("content"):
                return str(msg["content"])
    content = getattr(chunk, "content", None)
    if content:
        return str(content)
    text = getattr(chunk, "text", None)
    if text:
        return str(text)
    return ""


async def stream_llm_text(
    llm: Any,
    messages: List[dict],
) -> AsyncIterator[str]:
    """优先 astream；否则 ainvoke 后整块输出。"""
    astream = getattr(llm, "astream", None)
    if callable(astream):
        try:
            async for chunk in astream(messages):
                delta = extract_stream_delta(chunk)
                if delta:
                    yield delta
            return
        except TypeError:
            pass
        except NotImplementedError:
            pass

    response = await llm.ainvoke(messages)
    text = _extract_llm_text(response)
    if text:
        yield text
