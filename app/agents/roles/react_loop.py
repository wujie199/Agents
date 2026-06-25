"""L6 ReAct 循环：L1 快照 + L2 persist_turn + 工具调用。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.composition.run_context import RunContext
from core.composition.tool_dispatch import invoke_tool
from core.ports.memory import ToolCallRecord, TurnRecord

from agent_platform.memory.adapters.turn_buffer import TurnBuffer

from app.agents.orchestration.chat_config import ChatAgentConfig, load_chat_config
from app.agents.debug.debug_trace import agent_debug
from app.agents.memory.session_finalize import enrich_l1_before_finalize


def _extract_llm_text(response: Any) -> str:
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        return (getattr(msg, "content", None) or "").strip()
    if isinstance(response, str):
        return response.strip()
    return ""


def _resolve_turn_buffer(
    ctx: RunContext, turn_buffer: Optional[TurnBuffer]
) -> Optional[TurnBuffer]:
    return turn_buffer if turn_buffer is not None else ctx.turn_buffer


async def run_agent_turn(
    ctx: RunContext,
    user_message: str,
    *,
    model_role: str = "main_llm",
    persist: bool = True,
    turn_buffer: Optional[TurnBuffer] = None,
) -> str:
    """单轮对话：写入 user/assistant 到 L2，返回 assistant 文本。"""
    memory = ctx.require_memory()
    await memory.ensure_session(ctx.request)
    buf = _resolve_turn_buffer(ctx, turn_buffer)

    user_turn = TurnRecord(
        role="user",
        content=user_message,
        trace_id=ctx.request.trace_id,
    )
    if persist:
        if buf is not None:
            await buf.append(ctx.request, user_turn)
        else:
            await memory.persist_turn(ctx.request, user_turn)

    snap = memory.compose_prompt_snapshot(ctx.request)
    llm = ctx.get_model(model_role)
    response = await llm.ainvoke(
        [
            {"role": "system", "content": snap.memory_text},
            {"role": "user", "content": user_message},
        ]
    )
    assistant_text = _extract_llm_text(response)

    assistant_turn = TurnRecord(
        role="assistant",
        content=assistant_text,
        trace_id=ctx.request.trace_id,
    )
    if persist:
        if buf is not None:
            await buf.append(ctx.request, assistant_turn)
        else:
            await memory.persist_turn(ctx.request, assistant_turn)

    return assistant_text


async def execute_tool_calls(
    ctx: RunContext,
    tool_calls: List[Dict[str, Any]],
    *,
    persist: bool = True,
) -> List[Dict[str, Any]]:
    """执行工具列表并可选写入 L2 tool_calls。"""
    memory = ctx.require_memory()
    results: List[Dict[str, Any]] = []

    for call in tool_calls:
        name = call.get("name") or call.get("tool_name")
        args = call.get("args") or call.get("arguments") or {}
        if not name:
            continue

        start = time.time()
        status = "ok"
        result: Any = None
        try:
            result = await invoke_tool(ctx, name, args)
        except Exception as exc:
            status = "error"
            result = str(exc)

        latency_ms = int((time.time() - start) * 1000)
        if persist:
            await memory.persist_tool_call(
                ctx.request,
                ToolCallRecord(
                    tool_name=name,
                    args_hash=str(hash(frozenset(args.items())))[:16],
                    result_summary=str(result)[:500],
                    status=status,
                    latency_ms=latency_ms,
                ),
            )
        results.append(
            {"tool_name": name, "status": status, "result": result}
        )

    return results


async def end_agent_session(
    ctx: RunContext,
    *,
    status: str = "closed",
    flush_buffer: Optional[TurnBuffer] = None,
    checkpoint_state: Optional[dict] = None,
    chat_cfg: Optional[ChatAgentConfig] = None,
) -> dict:
    buf = flush_buffer if flush_buffer is not None else ctx.turn_buffer
    if buf is not None:
        await buf.flush()

    memory = ctx.memory
    mem_hash = ""
    if memory is not None and hasattr(memory, "get_snapshot_hash"):
        try:
            mem_hash = memory.get_snapshot_hash(
                ctx.request.tenant_id, ctx.request.user_id
            )
        except Exception:
            mem_hash = ""

    cp_state = checkpoint_state
    if cp_state is None:
        cp_state = {
            "session_id": ctx.request.session_id,
            "memory_snapshot_hash": mem_hash,
            "status": status,
        }
    elif mem_hash and "memory_snapshot_hash" not in cp_state:
        cp_state = {**cp_state, "memory_snapshot_hash": mem_hash}

    if ctx.checkpointer is not None:
        await ctx.checkpointer.save(
            thread_id=ctx.request.session_id,
            tenant_id=ctx.request.tenant_id,
            state=cp_state,
            session_id=ctx.request.session_id,
            metadata={"status": status},
        )

    agent_debug(
        "FINALIZE",
        "react_loop.end_agent_session:before",
        "end_session finalize 开始",
        {
            "status": status,
            "session_id": ctx.request.session_id,
            "tenant_id": ctx.request.tenant_id,
            "had_turn_buffer": buf is not None,
            "had_checkpoint": checkpoint_state is not None,
        },
    )
    cfg = chat_cfg or load_chat_config()
    l1_extract_pending = await enrich_l1_before_finalize(ctx, cfg)
    from app.agents.memory.memory_runtime_debug import (
        is_memory_runtime_debug,
        log_memory_runtime_status,
    )

    if is_memory_runtime_debug():
        await log_memory_runtime_status(ctx, event="finalize_before_end")
    summary = await ctx.require_memory().end_session(
        ctx.request, status=status, finalize=True
    )
    if not isinstance(summary, dict):
        summary = {}
    summary["l1_extract_pending"] = l1_extract_pending
    if isinstance(getattr(ctx, "extra", None), dict):
        ctx.extra["finalize_summary"] = summary
    agent_debug(
        "FINALIZE",
        "react_loop.end_agent_session:done",
        "finalize 完成",
        summary,
    )
    return summary
