# -*- coding: utf-8 -*-
"""RAG Query 改写：LLM 调用辅助（标准 messages 格式 + 响应解析）。"""

from __future__ import annotations

import logging
from typing import Any, List

_logger = logging.getLogger("rag.rewrite.llm_invoke")


def prompt_to_messages(prompt: str) -> List[dict[str, str]]:
    return [{"role": "user", "content": (prompt or "").strip()}]


def extract_llm_text(response: Any) -> str:
    if response is None:
        return ""
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        return (getattr(msg, "content", None) or "").strip()
    if hasattr(response, "content"):
        return str(response.content or "").strip()
    if isinstance(response, str):
        return response.strip()
    return str(response).strip()


async def invoke_llm_prompt(llm: Any, prompt: str) -> str:
    """调用 LLM 并返回文本；失败时抛出异常供上层降级。"""
    messages = prompt_to_messages(prompt)
    if hasattr(llm, "ainvoke"):
        response = await llm.ainvoke(messages)
    elif hasattr(llm, "invoke"):
        response = llm.invoke(messages)
    else:
        _logger.warning("LLM has no invoke/ainvoke method")
        return ""
    return extract_llm_text(response)
