# -*- coding: utf-8 -*-
"""ModelRegistry → LangChain BaseChatModel，供 create_react_agent 使用。"""

from __future__ import annotations

import json
import logging
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

from app.agents.prompts.llm_stream import stream_llm_text, stream_llm_chunks, extract_stream_chunk_reasoning
from app.agents.roles.react_loop import _extract_llm_text

_think_dbg = logging.getLogger("thinking_debug")


def _is_remove_message(msg: BaseMessage) -> bool:
    if msg.__class__.__name__ == "RemoveMessage":
        return True
    role = getattr(msg, "type", None) or ""
    return role == "remove"


def filter_messages_for_llm(messages: List[BaseMessage]) -> List[BaseMessage]:
    """去掉 LangGraph RemoveMessage，避免 API 收到非法 role=remove。"""
    return [m for m in messages if not _is_remove_message(m)]


def _message_to_dict(msg: BaseMessage) -> dict[str, Any] | None:
    if _is_remove_message(msg):
        return None
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


# ── 流式 tool_call 增量处理 ──


def _extract_stream_tool_call_deltas(chunk: Any) -> list[tuple[int, str, str, str]]:
    """从流式 chunk 提取 tool_call 增量。

    返回: [(index, id, name, arguments_delta), ...]
    OpenAI 流式格式: delta.tool_calls[i].function.arguments 逐 chunk 拼接。
    """
    if chunk is None:
        return []

    # OpenAI SDK streaming chunk
    if hasattr(chunk, "choices") and chunk.choices:
        delta = getattr(chunk.choices[0], "delta", None)
        if delta is not None:
            raw_tcs = getattr(delta, "tool_calls", None)
            if raw_tcs:
                results = []
                for tc in raw_tcs:
                    if isinstance(tc, dict):
                        idx = tc.get("index", 0)
                        tc_id = tc.get("id", "")
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        args_delta = fn.get("arguments", "")
                    else:
                        idx = getattr(tc, "index", 0)
                        tc_id = getattr(tc, "id", "") or ""
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "") or "" if fn else ""
                        args_delta = getattr(fn, "arguments", "") or "" if fn else ""
                    results.append((idx, tc_id, name, args_delta))
                return results

    # dict 格式
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            raw_tcs = delta.get("tool_calls") or []
            results = []
            for tc in raw_tcs:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index", 0)
                tc_id = tc.get("id", "")
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_delta = fn.get("arguments", "")
                results.append((idx, tc_id, name, args_delta))
            return results

    return []


def _merge_stream_tool_calls(
    accum: dict[int, dict], deltas: list[tuple[int, str, str, str]],
) -> None:
    """将 tool_call 增量合并到累积字典（就地修改）。"""
    for idx, tc_id, name, args_delta in deltas:
        if idx not in accum:
            accum[idx] = {"id": tc_id, "name": name, "args": args_delta}
        else:
            entry = accum[idx]
            if tc_id:
                entry["id"] = tc_id
            if name:
                entry["name"] = name
            entry["args"] = entry.get("args", "") + args_delta


def _accumulated_tc_to_lc(accum: dict[int, dict]) -> list[dict[str, Any]]:
    """将累积的 tool_call 增量转为 LangChain tool_calls 格式。"""
    result = []
    for idx in sorted(accum.keys()):
        entry = accum[idx]
        args_str = entry.get("args", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}
        result.append({
            "id": entry.get("id", ""),
            "name": entry.get("name", ""),
            "args": args,
        })
    return result


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
        dict_messages = [
            d for d in (_message_to_dict(m) for m in filter_messages_for_llm(messages)) if d
        ]
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

        # 提取 reasoning_content（推理模型思考过程）
        additional_kwargs: dict[str, Any] = {}
        reasoning = ""
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None) or ""
        if reasoning:
            additional_kwargs["reasoning_content"] = reasoning

        if tool_calls:
            message = AIMessage(
                content=text or "",
                tool_calls=tool_calls,
                additional_kwargs=additional_kwargs,
            )
        else:
            message = AIMessage(content=text, additional_kwargs=additional_kwargs)
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
            schemas = _tool_schemas(bound_tools)
            dict_messages = [
                d
                for d in (_message_to_dict(m) for m in filter_messages_for_llm(messages))
                if d
            ]
            llm = self.run_ctx.get_model(self.model_role)

            # 优先尝试流式输出（含 tools），逐 token 渲染 content/reasoning，
            # 同时累积 tool_call 增量，在末尾补充完整 tool_calls
            astream_fn = getattr(llm, "astream", None)
            if callable(astream_fn):
                try:
                    tc_accum: dict[int, dict] = {}  # {index: {id, name, args_str}}
                    has_content = False
                    _pcm_chunk_count = 0
                    _pcm_reasoning_chunks = 0

                    async for raw_chunk in astream_fn(dict_messages, tools=schemas):
                        content, reasoning = extract_stream_chunk_reasoning(raw_chunk)
                        # 提取 & 合并 tool_call 增量
                        tc_deltas = _extract_stream_tool_call_deltas(raw_chunk)
                        if tc_deltas:
                            _merge_stream_tool_calls(tc_accum, tc_deltas)

                        if content or reasoning:
                            _pcm_chunk_count += 1
                            if reasoning:
                                _pcm_reasoning_chunks += 1
                                if _pcm_reasoning_chunks == 1:
                                    _think_dbg.debug(
                                        "[THINK-4 PortChatModel.tools] 首个 reasoning chunk=%r",
                                        reasoning[:50],
                                    )
                            has_content = True
                            additional_kwargs = {}
                            if reasoning:
                                additional_kwargs["reasoning_content"] = reasoning
                            yield ChatGenerationChunk(message=AIMessageChunk(
                                content=content,
                                additional_kwargs=additional_kwargs,
                            ))
                            if run_manager:
                                await run_manager.on_llm_new_token(content or reasoning)

                    _think_dbg.debug(
                        "[THINK-4 PortChatModel.tools] 流式结束: chunks=%d reasoning_chunks=%d tool_calls=%d",
                        _pcm_chunk_count, _pcm_reasoning_chunks, len(tc_accum),
                    )

                    # 流式结束：补充累积的 tool_calls
                    if tc_accum:
                        lc_tool_calls = _accumulated_tc_to_lc(tc_accum)
                        yield ChatGenerationChunk(message=AIMessageChunk(
                            content="",
                            tool_calls=lc_tool_calls,
                        ))
                    elif not has_content:
                        # 无内容也无 tool_calls — 可能是空响应，忽略
                        pass
                    return
                except (TypeError, NotImplementedError, AttributeError):
                    pass  # 降级到非流式

            # 降级：非流式整块返回
            result = await self._agenerate(messages, stop, run_manager, **kwargs)
            for gen in result.generations:
                msg = gen.message
                chunk_msg = AIMessageChunk(
                    content=msg.content,
                    tool_calls=getattr(msg, "tool_calls", None),
                    additional_kwargs=getattr(msg, "additional_kwargs", {}) or {},
                )
                yield ChatGenerationChunk(message=chunk_msg)
            return

        dict_messages = [
            d for d in (_message_to_dict(m) for m in filter_messages_for_llm(messages)) if d
        ]
        llm = self.run_ctx.get_model(self.model_role)
        _notools_reasoning_chunks = 0
        async for content_delta, reasoning_delta in stream_llm_chunks(
            llm, dict_messages
        ):
            additional_kwargs = {}
            if reasoning_delta:
                _notools_reasoning_chunks += 1
                if _notools_reasoning_chunks == 1:
                    _think_dbg.debug(
                        "[THINK-4 PortChatModel.notools] 首个 reasoning chunk=%r",
                        reasoning_delta[:50],
                    )
                additional_kwargs["reasoning_content"] = reasoning_delta
            chunk = AIMessageChunk(
                content=content_delta,
                additional_kwargs=additional_kwargs,
            )
            yield ChatGenerationChunk(message=chunk)
            token_text = content_delta or reasoning_delta
            if run_manager and token_text:
                await run_manager.on_llm_new_token(token_text)
        _think_dbg.debug(
            "[THINK-4 PortChatModel.notools] 流式结束: reasoning_chunks=%d",
            _notools_reasoning_chunks,
        )
