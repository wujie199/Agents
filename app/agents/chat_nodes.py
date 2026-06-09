# -*- coding: utf-8 -*-
"""对话图节点：记忆 L1/L2 + RAG 准备与 L2 持久化（供 LangGraph / chat_turn 共用）。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, List, Optional

from core.composition.run_context import RunContext
from core.domain.evidence import DegradedReason, EvidenceBundle
from core.ports.memory import TurnRecord

from app.agents.text_sanitize import (
    sanitize_memory_text_for_chat,
    sanitize_turn_content,
    strip_model_reasoning,
)
from app.agents.chat_config import ChatAgentConfig
from app.agents.context_builder import prepare_session_context
from app.agents.retrieval_router import build_retrieval_plan_async
from app.agents.debug_trace import agent_debug, summarize_messages
from app.agents.memory_runtime_debug import (
    clear_layer_triggers,
    log_layer_trigger_summary,
    perf_await,
    perf_mark,
    perf_sync,
    trace_layer_trigger,
)
from app.agents.context_builder import is_name_intro_query
from app.agents.prompt_builder import (
    PROFILE_NAME_HINT,
    RECALL_TOOL_HINT,
    build_chat_messages,
    filter_evidence_bundle,
    format_evidence_bundle,
)


async def fetch_turn_history(
    memory: Any,
    request: Any,
    *,
    max_turns: int,
    turn_buffer: Any = None,
) -> List[dict]:
    if max_turns <= 0:
        return []
    rows = await memory.list_turns(request, limit=max_turns * 2 + 10)
    if turn_buffer is not None and hasattr(turn_buffer, "pending_turns_for"):
        rows = [*rows, *turn_buffer.pending_turns_for(request)]
    history: List[dict] = []
    for row in rows:
        role = row.get("role")
        if role in ("user", "assistant"):
            content = sanitize_turn_content(row.get("content"), role=role)
            if content:
                history.append({**row, "content": content})
    if len(history) > max_turns * 2:
        history = history[-(max_turns * 2) :]
    return history


async def retrieve_rag_bundle(
    ctx: RunContext,
    query: str,
    *,
    plan: Optional[dict] = None,
) -> EvidenceBundle:
    # #region agent log
    agent_debug(
        "RAG-C",
        "chat_nodes.retrieve_rag_bundle:entry",
        "RAG 检索开始",
        {
            "query_preview": query[:120],
            "rag_port": ctx.rag is not None,
            "tenant_id": getattr(ctx.request, "tenant_id", None),
            "session_id": getattr(ctx.request, "session_id", None),
        },
    )
    # #endregion
    if ctx.rag is None:
        trace_layer_trigger(ctx, "RAG", "route_and_retrieve", False, "rag_not_configured")
        bundle = EvidenceBundle.empty_bundle(
            DegradedReason.VECTOR_UNAVAILABLE,
            "rag_not_configured",
        )
        # #region agent log
        agent_debug(
            "RAG-C",
            "chat_nodes.retrieve_rag_bundle:no_port",
            "RAGPort 未注入",
            {"empty": True, "error_code": "rag_not_configured"},
        )
        # #endregion
        return bundle
    rag_tenant = (getattr(ctx, "extra", None) or {}).get("rag_tenant_id")
    rag_request = (
        replace(ctx.request, tenant_id=rag_tenant)
        if rag_tenant and rag_tenant != ctx.request.tenant_id
        else ctx.request
    )
    # #region agent log
    agent_debug(
        "RAG-C",
        "chat_nodes.retrieve_rag_bundle:tenant",
        "RAG 检索 tenant",
        {
            "memory_tenant": ctx.request.tenant_id,
            "rag_tenant": rag_request.tenant_id,
            "chroma_dir": (getattr(ctx, "extra", None) or {}).get("rag_chroma_dir"),
        },
    )
    # #endregion
    import time as _time

    _rag_t0 = _time.perf_counter()
    bundle = await ctx.rag.route_and_retrieve(query, rag_request, plan=plan)
    perf_mark(
        ctx,
        "RAG.route_and_retrieve",
        (_time.perf_counter() - _rag_t0) * 1000,
        evidence_count=len(bundle.evidences or []),
    )
    # #region agent log
    agent_debug(
        "RAG-D",
        "chat_nodes.retrieve_rag_bundle:result",
        "RAG 检索完成",
        {
            "empty": bundle.empty,
            "degraded": bundle.is_degraded(),
            "degraded_reason": (
                bundle.degraded_reason.value if bundle.degraded_reason else None
            ),
            "error_code": bundle.error_code,
            "evidence_count": len(bundle.evidences or []),
            "evidences": [
                {
                    "id": ev.id,
                    "score": round(ev.score, 4),
                    "citation": ev.citation,
                    "content_preview": (ev.content or "")[:150],
                }
                for ev in (bundle.evidences or [])[:5]
            ],
            "plan_mode": (bundle.plan or {}).get("mode"),
        },
    )
    # #endregion
    return bundle


async def build_turn_messages(
    ctx: RunContext,
    user_message: str,
    cfg: ChatAgentConfig,
    *,
    enable_rag: bool,
    rag_plan: Optional[dict] = None,
) -> tuple[list[dict[str, str]], int, bool, str]:
    """返回 (llm_messages, evidence_count, rag_empty, memory_snapshot_hash)。"""
    clear_layer_triggers(ctx)
    memory = ctx.require_memory()
    await perf_await(ctx, "L2.ensure_session", memory.ensure_session(ctx.request))
    trace_layer_trigger(
        ctx, "L2", "ensure_session", True, "session_row_exists_or_created"
    )

    snap = perf_sync(ctx, "L1.compose_snapshot", memory.compose_prompt_snapshot, ctx.request)
    trace_layer_trigger(
        ctx,
        "L1",
        "compose_prompt_snapshot",
        True,
        "inject_hot_memory",
        data={
            "hash": snap.hash,
            "chars": len(snap.memory_text or ""),
        },
    )
    # #region agent log
    agent_debug(
        "MEM-L1",
        "chat_nodes.build_turn_messages:l1",
        "L1 热记忆快照",
        {
            "hash": snap.hash,
            "memory_chars": len(snap.memory_text or ""),
            "memory_preview": (snap.memory_text or "")[:400],
        },
    )
    # #endregion

    history = await perf_await(
        ctx,
        "L2.load_history",
        fetch_turn_history(
            memory,
            ctx.request,
            max_turns=cfg.max_history_turns,
            turn_buffer=ctx.turn_buffer,
        ),
        history_count_hint=cfg.max_history_turns,
    )
    trace_layer_trigger(
        ctx,
        "L2",
        "load_turn_history",
        True,
        "list_turns+buffer",
        data={
            "history_count": len(history),
            "max_history_turns": cfg.max_history_turns,
        },
    )
    retrieval_plan = await perf_await(
        ctx,
        "ROUTER.build_plan",
        build_retrieval_plan_async(
            user_message,
            cfg,
            enable_rag=enable_rag,
            models=ctx.models,
            ctx=ctx,
        ),
    )
    if isinstance(getattr(ctx, "extra", None), dict):
        ctx.extra["retrieval_intent"] = retrieval_plan.intent

    parallel_rag = (
        enable_rag
        and retrieval_plan.run_rag
        and retrieval_plan.intent != "recall"
    )
    bundle: EvidenceBundle | None = None
    if parallel_rag:
        session_ctx, bundle = await asyncio.gather(
            perf_await(
                ctx,
                "CTX.prepare_session_context",
                prepare_session_context(
                    ctx,
                    history,
                    user_message,
                    cfg,
                    retrieval_plan=retrieval_plan,
                ),
                intent=retrieval_plan.intent,
            ),
            perf_await(
                ctx,
                "RAG.retrieve_bundle",
                retrieve_rag_bundle(ctx, user_message, plan=rag_plan),
            ),
        )
    else:
        session_ctx = await perf_await(
            ctx,
            "CTX.prepare_session_context",
            prepare_session_context(
                ctx, history, user_message, cfg, retrieval_plan=retrieval_plan
            ),
            intent=retrieval_plan.intent,
        )
    trimmed_history = session_ctx.trimmed_history
    session_context = session_ctx.extra_text
    agent_debug(
        "RETRIEVAL",
        "chat_nodes.build_turn_messages:plan",
        "检索编排计划",
        retrieval_plan.to_debug_dict(),
    )
    # #region agent log
    agent_debug(
        "MEM-L2-HIST",
        "chat_nodes.build_turn_messages:history",
        "L2 会话历史",
        {
            "history_count": len(history),
            "trimmed_count": len(trimmed_history),
            "session_context_chars": len(session_context or ""),
            "max_history_turns": cfg.max_history_turns,
            "turns": [
                {
                    "role": t.get("role"),
                    "chars": len(t.get("content") or ""),
                    "preview": (t.get("content") or "")[:80],
                }
                for t in trimmed_history[-6:]
            ],
        },
    )
    # #endregion

    evidence_text = ""
    run_rag = enable_rag and retrieval_plan.run_rag
    if (
        run_rag
        and cfg.recall_skip_rag_when_prefetch_hit
        and retrieval_plan.intent == "recall"
        and session_ctx.recall_prefetch_hit
    ):
        run_rag = False
        trace_layer_trigger(
            ctx, "RAG", "retrieve", False, "recall_prefetch_hit"
        )
        agent_debug(
            "RAG-SKIP",
            "chat_nodes.build_turn_messages:recall_prefetch_hit",
            "回忆预检索已命中，跳过 RAG",
            {"recall_chars": len(session_ctx.recall_prefetch or "")},
        )
    if (
        run_rag
        and cfg.recall_skip_rag_when_prefetch_miss
        and retrieval_plan.intent == "recall"
        and retrieval_plan.run_session_search
        and not session_ctx.recall_prefetch_hit
    ):
        run_rag = False
        trace_layer_trigger(
            ctx, "RAG", "retrieve", False, "recall_prefetch_miss"
        )
        agent_debug(
            "RAG-SKIP",
            "chat_nodes.build_turn_messages:recall_prefetch_miss",
            "回忆预检索无结果，跳过 RAG",
            {"intent": retrieval_plan.intent},
        )
    if run_rag and bundle is None:
        trace_layer_trigger(
            ctx, "RAG", "retrieve", True, "plan_or_gating",
            data={"intent": retrieval_plan.intent},
        )
        bundle = await perf_await(
            ctx,
            "RAG.retrieve_bundle",
            retrieve_rag_bundle(ctx, user_message, plan=rag_plan),
        )
    elif run_rag and bundle is not None:
        trace_layer_trigger(
            ctx,
            "RAG",
            "retrieve",
            True,
            "parallel_with_context",
            data={"intent": retrieval_plan.intent},
        )
    if run_rag and bundle is not None:
        bundle = filter_evidence_bundle(
            bundle,
            min_score=cfg.rag_min_score,
            min_keep=cfg.rag_min_keep,
        )
        evidence_text = format_evidence_bundle(
            bundle,
            max_chars=cfg.max_evidence_chars,
            max_items=cfg.max_evidence_items,
            min_score=cfg.rag_min_score,
            min_keep=cfg.rag_min_keep,
            strict_grounding=cfg.evidence_strict_grounding,
        )
    elif enable_rag and retrieval_plan.skip_rag_reason:
        trace_layer_trigger(
            ctx,
            "RAG",
            "retrieve",
            False,
            retrieval_plan.skip_rag_reason or "orchestration",
        )
        agent_debug(
            "RAG-SKIP",
            "chat_nodes.build_turn_messages:rag_orchestrated",
            "RAG 编排跳过",
            {
                "reason": retrieval_plan.skip_rag_reason,
                "intent": retrieval_plan.intent,
                "query_preview": user_message[:80],
            },
        )
    elif enable_rag and not retrieval_plan.run_rag:
        trace_layer_trigger(
            ctx,
            "RAG",
            "retrieve",
            False,
            f"intent={retrieval_plan.intent}",
        )
        agent_debug(
            "RAG-SKIP",
            "chat_nodes.build_turn_messages:rag_gated",
            "RAG 门控跳过（寒暄/过短/编排）",
            {
                "query_preview": user_message[:80],
                "intent": retrieval_plan.intent,
            },
        )
    else:
        trace_layer_trigger(ctx, "RAG", "retrieve", False, "enable_rag=false")
        # #region agent log
        agent_debug(
            "RAG-SKIP",
            "chat_nodes.build_turn_messages:rag_off",
            "RAG 已关闭 (--no-rag)",
            {"enable_rag": False},
        )
        # #endregion

    # #region agent log
    agent_debug(
        "RAG-D",
        "chat_nodes.build_turn_messages:evidence_text",
        "RAG 注入 prompt 的 evidence 块",
        {
            "evidence_chars": len(evidence_text),
            "evidence_preview": evidence_text[:300] if evidence_text else "",
        },
    )
    # #endregion

    tool_hints = ""
    if (
        cfg.enable_memory_tools
        and cfg.recall_tool_hint_on_miss
        and retrieval_plan.intent == "recall"
        and retrieval_plan.run_session_search
        and not session_ctx.recall_prefetch_hit
    ):
        tool_hints = RECALL_TOOL_HINT
    elif (
        cfg.enable_memory_tools
        and retrieval_plan.intent == "profile"
        and is_name_intro_query(user_message)
    ):
        tool_hints = PROFILE_NAME_HINT

    messages = build_chat_messages(
        memory_system=sanitize_memory_text_for_chat(snap.memory_text),
        user_message=user_message,
        evidence_text=evidence_text,
        session_context_text=session_context,
        history=trimmed_history,
        tool_hints=tool_hints,
    )
    ev_count = len(bundle.evidences) if bundle and bundle.evidences else 0
    # #region agent log
    agent_debug(
        "PROMPT",
        "chat_nodes.build_turn_messages:assembled",
        "最终 LLM messages 组装",
        summarize_messages(messages),
    )
    # #endregion
    log_layer_trigger_summary(
        ctx,
        user_message=user_message,
        extra={
            "evidence_count": ev_count,
            "rag_empty": bundle.empty if bundle else True,
            "evidence_chars": len(evidence_text),
        },
    )
    # #region agent log
    _summary_part = ""
    if session_context and "【更早对话摘要】" in session_context:
        _si = session_context.find("【更早对话摘要】")
        _rest = session_context[_si:]
        _next = _rest.find("\n\n【", len("【更早对话摘要】"))
        _summary_part = _rest[:_next] if _next > 0 else _rest
    from app.agents.memory_runtime_debug import chat_config_verify_snapshot, trace_write

    trace_write(
        hypothesis_id="VERIFY-R2",
        location="chat_nodes.build_turn_messages:verify",
        message="round2 per-turn verification",
        data={
            "query_preview": (user_message or "")[:80],
            "intent": retrieval_plan.intent,
            "channels": retrieval_plan.to_debug_dict().get("channels"),
            "run_session_search": retrieval_plan.run_session_search,
            "run_rag_planned": retrieval_plan.run_rag,
            "run_rag_actual": bool(run_rag and bundle),
            "run_l4": retrieval_plan.run_l4,
            "recall_prefetch_hit": session_ctx.recall_prefetch_hit,
            "has_session_search_block": any(
                marker in (session_context or "")
                for marker in (
                    "【会话回忆检索】",
                    "【跨会话回忆检索】",
                    "【会话相关检索】",
                    "【跨会话相关】",
                )
            ),
            "has_recall_block": any(
                marker in (session_context or "")
                for marker in (
                    "【会话回忆检索】",
                    "【跨会话回忆检索】",
                    "【会话相关检索】",
                    "【跨会话相关】",
                )
            ),
            "has_l4_block": "【外部画像 L4】" in (session_context or ""),
            "summary_has_assistant": "assistant:" in _summary_part.lower(),
            "summary_user_only_cfg": cfg.rolling_summary_user_only,
            "has_grounding_rules": "【回答约束】" in evidence_text,
            "has_tool_hints": bool(tool_hints),
            "config": chat_config_verify_snapshot(cfg),
        },
        run_id=getattr(ctx.request, "session_id", None) or "default",
        force=True,
    )
    # #endregion
    return messages, ev_count, bundle.empty if bundle else True, snap.hash


async def persist_user_and_assistant(
    ctx: RunContext,
    *,
    user_message: str,
    assistant_text: str,
    persist: bool = True,
    turn_buffer: Any = None,
    chat_cfg: ChatAgentConfig | None = None,
) -> None:
    if not persist:
        return
    memory = ctx.require_memory()
    trace = ctx.request.trace_id
    buf = turn_buffer if turn_buffer is not None else ctx.turn_buffer

    assistant_text = strip_model_reasoning(assistant_text)
    user_turn = TurnRecord(role="user", content=user_message, trace_id=trace)
    assistant_turn = TurnRecord(
        role="assistant", content=assistant_text, trace_id=trace
    )

    if buf is not None:
        await buf.append(ctx.request, user_turn)
        await buf.append(ctx.request, assistant_turn)
        via = "turn_buffer"
        cfg = chat_cfg
        if cfg is not None and cfg.interactive_flush_buffer:
            await buf.flush()
            via = "turn_buffer_flush"
        trace_layer_trigger(ctx, "L2", "persist_turn", True, via)
    else:
        await memory.persist_turn(ctx.request, user_turn)
        await memory.persist_turn(ctx.request, assistant_turn)
        via = "persist_turn"
        trace_layer_trigger(ctx, "L2", "persist_turn", True, via)

    rows = await memory.list_turns(ctx.request, limit=20)
    pending = 0
    if buf is not None and hasattr(buf, "pending_turns_for"):
        pending = len(buf.pending_turns_for(ctx.request))
    # #region agent log
    agent_debug(
        "MEM-L2-PERSIST",
        "chat_nodes.persist_user_and_assistant",
        "L2 写入完成",
        {
            "via": via,
            "user_chars": len(user_message),
            "assistant_chars": len(assistant_text),
            "session_id": ctx.request.session_id,
            "total_turns_after": len(rows),
            "pending_in_buffer": pending,
            "last_turns": [
                {
                    "role": r.get("role"),
                    "chars": len(r.get("content") or ""),
                }
                for r in rows[-4:]
            ],
        },
    )
    # #endregion
