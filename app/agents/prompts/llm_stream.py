# -*- coding: utf-8 -*-
"""LLM 流式输出适配。"""

from __future__ import annotations

from typing import Any, AsyncIterator, List, Tuple

from app.agents.roles.react_loop import _extract_llm_text


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


def extract_stream_chunk_reasoning(chunk: Any) -> Tuple[str, str]:
    """从 astream chunk 提取 (content_delta, reasoning_delta)。

    支持:
      - OpenAI SDK 原生 chunk: choice.delta.content / .reasoning_content
      - DashScope 兼容 chunk: message.reasoning_content
      - provider 自定义 astream yield (content, reasoning) 元组
    """
    # provider 层 astream 直接 yield 元组
    if isinstance(chunk, tuple) and len(chunk) == 2:
        return (str(chunk[0] or ""), str(chunk[1] or ""))

    if chunk is None:
        return ("", "")

    # OpenAI SDK streaming chunk
    if hasattr(chunk, "choices") and chunk.choices:
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None) or ""
            reasoning = getattr(delta, "reasoning_content", None) or ""
            return (str(content), str(reasoning))
        # 非流式 message（fallback）
        msg = getattr(choice, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None) or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            return (str(content), str(reasoning))

    # dict 格式
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content", "")
            reasoning = delta.get("reasoning_content", "")
            if content or reasoning:
                return (str(content), str(reasoning))
            msg = choices[0].get("message") or {}
            return (str(msg.get("content", "")), str(msg.get("reasoning_content", "")))

    # fallback: 只取 content
    content = getattr(chunk, "content", None)
    if content:
        return (str(content), "")
    return ("", "")


async def stream_llm_text(
    llm: Any,
    messages: List[dict],
) -> AsyncIterator[str]:
    """优先 astream；否则 ainvoke 后整块输出。仅 yield content。"""
    astream = getattr(llm, "astream", None)
    if callable(astream):
        try:
            async for chunk in astream(messages):
                content, reasoning = extract_stream_chunk_reasoning(chunk)
                if content:
                    yield content
            return
        except TypeError:
            pass
        except NotImplementedError:
            pass

    response = await llm.ainvoke(messages)
    text = _extract_llm_text(response)
    if text:
        yield text


async def stream_llm_chunks(
    llm: Any,
    messages: List[dict],
) -> AsyncIterator[Tuple[str, str]]:
    """流式输出 (content_delta, reasoning_delta)。

    优先 astream；否则 ainvoke 后整块返回。
    """
    astream = getattr(llm, "astream", None)
    if callable(astream):
        try:
            async for chunk in astream(messages):
                content, reasoning = extract_stream_chunk_reasoning(chunk)
                if content or reasoning:
                    yield (content, reasoning)
            return
        except TypeError:
            pass
        except NotImplementedError:
            pass

    response = await llm.ainvoke(messages)
    # 尝试从完整响应中提取 reasoning_content
    reasoning = ""
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or ""
    text = _extract_llm_text(response)
    if text or reasoning:
        yield (text, reasoning)
