# -*- coding: utf-8 -*-
"""ModelRegistry → LangChain BaseChatModel，供 create_react_agent 使用。"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.utils.function_calling import convert_to_openai_tool

from core.composition.run_context import RunContext

from app.agents.llm_stream import stream_llm_text
from app.agents.react_loop import _extract_llm_text


def _message_to_dict(msg: BaseMessage) -> dict[str, Any]:
    role = getattr(msg, "type", None) or "user"
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    elif role == "tool":
        return {
            "role": "tool",
            "content": str(msg.content or ""),
            "tool_call_id": getattr(msg, "tool_call_id", ""),
        }
    content = msg.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        content = "\n".join(parts)
    out: dict[str, Any] = {"role": role, "content": str(content or "")}
    if isinstance(msg, AIMessage) and msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def _tool_schemas(tools: Sequence[Any]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for tool in tools:
        try:
            schemas.append(convert_to_openai_tool(tool))
        except Exception:
            name = getattr(tool, "name", None) or str(tool)
            schemas.append(
                {
                    "type": "function",
                    "function": {"name": name, "parameters": {"type": "object"}},
                }
            )
    return schemas


def _lc_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    if not raw_calls:
        return []
    out: list[dict[str, Any]] = []
    for tc in raw_calls:
        if isinstance(tc, dict):
            if "function" in tc:
                fn = tc["function"]
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                out.append(
                    {
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "args": args or {},
                    }
                )
            else:
                out.append(
                    {
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    }
                )
            continue
        fn = getattr(tc, "function", None)
        if fn is not None:
            args_raw = getattr(fn, "arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            out.append(
                {
                    "id": getattr(tc, "id", ""),
                    "name": getattr(fn, "name", ""),
                    "args": args or {},
                }
            )
    return out


def _extract_tool_calls_from_response(response: Any) -> list[dict[str, Any]]:
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        return _lc_tool_calls(getattr(msg, "tool_calls", None))
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            return _lc_tool_calls(msg.get("tool_calls"))
    return []


class PortChatModel(BaseChatModel):
    """将 RunContext.get_model(role).ainvoke(dict messages) 适配为 LangChain ChatModel。"""

    model_role: str = "main_llm"

    def __init__(
        self,
        run_ctx: RunContext,
        model_role: str = "main_llm",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_role=model_role, **kwargs)
        object.__setattr__(self, "_run_ctx", run_ctx)
        object.__setattr__(self, "_bound_tools", [])

    @property
    def run_ctx(self) -> RunContext:
        return object.__getattribute__(self, "_run_ctx")

    @property
    def _llm_type(self) -> str:
        return "port_chat_model"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> "PortChatModel":
        bound = PortChatModel(run_ctx=self.run_ctx, model_role=self.model_role)
        object.__setattr__(bound, "_bound_tools", list(tools))
        object.__setattr__(bound, "_tool_choice", tool_choice)
        return bound

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import asyncio

        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        dict_messages = [_message_to_dict(m) for m in messages]
        llm = self.run_ctx.get_model(self.model_role)
        bound_tools = object.__getattribute__(self, "_bound_tools")

        response: Any
        if bound_tools:
            schemas = _tool_schemas(bound_tools)
            try:
                response = await llm.ainvoke(dict_messages, tools=schemas)
            except TypeError:
                response = await llm.ainvoke(dict_messages)
        else:
            response = await llm.ainvoke(dict_messages)

        text = _extract_llm_text(response)
        tool_calls = _extract_tool_calls_from_response(response)
        if tool_calls:
            message = AIMessage(content=text or "", tool_calls=tool_calls)
        else:
            message = AIMessage(content=text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        bound_tools = object.__getattribute__(self, "_bound_tools")
        if bound_tools:
            result = await self._agenerate(messages, stop, run_manager, **kwargs)
            for gen in result.generations:
                yield ChatGenerationChunk(message=gen.message)
            return

        dict_messages = [_message_to_dict(m) for m in messages]
        llm = self.run_ctx.get_model(self.model_role)
        async for delta in stream_llm_text(llm, dict_messages):
            chunk = AIMessageChunk(content=delta)
            yield ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(delta)
