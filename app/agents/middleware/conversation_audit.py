# -*- coding: utf-8
"""ConversationAuditMiddleware：prepare/persist 汇总 turn 级企业审计包。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.agents.middleware.audit_content import (
    AuditContentMode,
    apply_audit_content,
    normalize_audit_content_mode,
    redact_preview,
)
from app.agents.middleware.audit_persist import persist_conversation_turn_event

_logger = logging.getLogger("app.agents.middleware.conversation_audit")

_PREPARE_STASH_KEY = "_turn_audit_prepare"


class ConversationAuditMiddleware:
    """在 prepare 暂存检索摘要，persist 写入 turn 级 NDJSON。"""

    def __init__(
        self,
        *,
        persist: bool = False,
        audit_log_dir: str = "data/audit",
        audit_content: str = "redacted",
        audit_include_retrieval: bool = True,
        audit_include_tools: bool = True,
        audit_max_content_chars: int = 8000,
    ) -> None:
        self._persist = persist
        self._audit_log_dir = audit_log_dir
        self._content_mode: AuditContentMode = normalize_audit_content_mode(
            audit_content
        )
        self._include_retrieval = audit_include_retrieval
        self._include_tools = audit_include_tools
        self._max_chars = audit_max_content_chars

    @property
    def name(self) -> str:
        return "conversation_audit"

    async def on_enter(
        self,
        node_name: str,
        state: Any,
        config: Any,
    ) -> Dict[str, Any]:
        return {}

    async def on_exit(
        self,
        node_name: str,
        state: Any,
        config: Any,
        result: Any,
        *,
        error: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx_extra = extra or {}
        configurable = (config or {}).get("configurable") or {}
        run_ctx = configurable.get("run_ctx")

        if node_name == "prepare":
            _stash_prepare(run_ctx, state, result)
            return

        if node_name != "persist":
            return

        entry = build_conversation_turn_entry(
            state=state if isinstance(state, dict) else {},
            result=result if isinstance(result, dict) else {},
            run_ctx=run_ctx,
            ctx_extra=ctx_extra,
            error=error,
            content_mode=self._content_mode,
            include_retrieval=self._include_retrieval,
            include_tools=self._include_tools,
            max_content_chars=self._max_chars,
        )
        if run_ctx is not None and isinstance(getattr(run_ctx, "extra", None), dict):
            turns = run_ctx.extra.setdefault("conversation_turn_audit", [])
            if isinstance(turns, list):
                turns.append(dict(entry))

        _logger.info("conversation_audit turn %s", entry.get("turn_id"))
        if self._persist:
            persist_conversation_turn_event(entry, self._audit_log_dir)


def _stash_prepare(run_ctx: Any, state: Any, result: Any) -> None:
    if run_ctx is None or not isinstance(getattr(run_ctx, "extra", None), dict):
        return
    src = result if isinstance(result, dict) else {}
    st = state if isinstance(state, dict) else {}
    run_ctx.extra[_PREPARE_STASH_KEY] = {
        "user_input": src.get("user_input") or st.get("user_input") or "",
        "evidence_count": src.get("evidence_count", st.get("evidence_count")),
        "rag_empty": src.get("rag_empty", st.get("rag_empty")),
        "memory_snapshot_hash": src.get("memory_snapshot_hash")
        or st.get("memory_snapshot_hash")
        or "",
        "evidences_summary": list(src.get("evidences_summary") or st.get("evidences_summary") or []),
        "memory_summary": dict(src.get("memory_summary") or st.get("memory_summary") or {}),
        "retrieval_intent": src.get("retrieval_intent") or st.get("retrieval_intent") or "",
        "memory_path": src.get("memory_path") or st.get("memory_path") or "",
    }


def build_conversation_turn_entry(
    *,
    state: dict[str, Any],
    result: dict[str, Any],
    run_ctx: Any,
    ctx_extra: dict[str, Any],
    error: Optional[Exception],
    content_mode: AuditContentMode,
    include_retrieval: bool,
    include_tools: bool,
    max_content_chars: int,
) -> dict[str, Any]:
    tenant_id = user_id = session_id = trace_id = turn_id = ""
    extra = getattr(run_ctx, "extra", None) if run_ctx is not None else None
    if run_ctx is not None:
        req = getattr(run_ctx, "request", None)
        if req is not None:
            tenant_id = str(getattr(req, "tenant_id", "") or "")
            user_id = str(getattr(req, "user_id", "") or "")
            session_id = str(getattr(req, "session_id", "") or "")
            trace_id = str(getattr(req, "trace_id", "") or "")

    if isinstance(extra, dict):
        turn_id = str(extra.get("_active_turn_id") or "")
        if not trace_id:
            trace_id = str(ctx_extra.get("trace_id") or "")

    prepare_stash = {}
    if isinstance(extra, dict):
        raw_stash = extra.get(_PREPARE_STASH_KEY)
        if isinstance(raw_stash, dict):
            prepare_stash = raw_stash

    user_input = (
        state.get("user_input")
        or prepare_stash.get("user_input")
        or ""
    )
    assistant_text = (
        result.get("assistant_text")
        or state.get("assistant_text")
        or ""
    )

    entry: dict[str, Any] = {
        "event_type": "conversation.turn",
        "trace_id": trace_id or ctx_extra.get("trace_id", "unknown"),
        "turn_id": turn_id or None,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "session_id": session_id,
        "content_policy": content_mode,
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_input": apply_audit_content(
            str(user_input), content_mode, max_chars=max_content_chars
        ),
        "assistant_text": apply_audit_content(
            str(assistant_text), content_mode, max_chars=max_content_chars
        ),
        "error": str(error) if error else None,
    }

    if include_retrieval:
        entry["prepare"] = _build_prepare_section(state, prepare_stash)
        entry["memory"] = _build_memory_section(state, extra if isinstance(extra, dict) else {})

    if include_tools and isinstance(extra, dict):
        entry["tools"] = _summarize_agent_events(extra.get("agent_events"))

    node_audit = []
    if isinstance(extra, dict):
        raw_nodes = extra.get("turn_audit")
        if isinstance(raw_nodes, list):
            node_audit = [
                {
                    "node": n.get("node"),
                    "duration_ms": n.get("duration_ms"),
                    "error": n.get("error"),
                }
                for n in raw_nodes
                if isinstance(n, dict)
            ]
    entry["nodes"] = node_audit

    return entry


def _build_prepare_section(
    state: dict[str, Any], prepare_stash: dict[str, Any]
) -> dict[str, Any]:
    evidences = prepare_stash.get("evidences_summary") or state.get("evidences_summary") or []
    mem_summary = prepare_stash.get("memory_summary") or state.get("memory_summary") or {}
    safe_evidences: List[dict[str, Any]] = []
    for ev in evidences[:10]:
        if not isinstance(ev, dict):
            continue
        safe_evidences.append(
            {
                "id": ev.get("id"),
                "score": ev.get("score"),
                "citation": redact_preview(str(ev.get("citation") or ""), 120),
                "content_preview": redact_preview(str(ev.get("content_preview") or ""), 200),
            }
        )
    safe_mem = {}
    if isinstance(mem_summary, dict):
        for key in (
            "recall_hit",
            "skill_hit",
            "l4_hit",
            "retrieval_intent",
            "memory_path",
            "l0_applied",
            "recall_strategy",
            "recall_scope",
            "intent_source",
            "rules_version",
        ):
            if key in mem_summary:
                safe_mem[key] = mem_summary.get(key)
        if mem_summary.get("recall_preview"):
            safe_mem["recall_preview"] = redact_preview(
                str(mem_summary.get("recall_preview")), 300
            )
    return {
        "evidence_count": prepare_stash.get("evidence_count", state.get("evidence_count")),
        "rag_empty": prepare_stash.get("rag_empty", state.get("rag_empty")),
        "memory_snapshot_hash": prepare_stash.get("memory_snapshot_hash")
        or state.get("memory_snapshot_hash")
        or "",
        "retrieval_intent": prepare_stash.get("retrieval_intent")
        or state.get("retrieval_intent")
        or "",
        "memory_path": prepare_stash.get("memory_path") or state.get("memory_path") or "",
        "recall_strategy": mem_summary.get("recall_strategy")
        or prepare_stash.get("recall_strategy")
        or state.get("recall_strategy")
        or "",
        "recall_scope": mem_summary.get("recall_scope")
        or prepare_stash.get("recall_scope")
        or state.get("recall_scope")
        or "",
        "intent_source": mem_summary.get("intent_source")
        or prepare_stash.get("intent_source")
        or state.get("intent_source")
        or "",
        "rules_version": mem_summary.get("rules_version")
        or prepare_stash.get("rules_version")
        or state.get("rules_version")
        or "",
        "evidences_summary": safe_evidences,
        "memory_summary": safe_mem,
    }


def _build_memory_section(
    state: dict[str, Any], extra: dict[str, Any]
) -> dict[str, Any]:
    turn_decision = extra.get("turn_decision")
    layer_triggers = extra.get("layer_triggers")
    triggers_out: List[dict[str, Any]] = []
    if isinstance(layer_triggers, list):
        for t in layer_triggers[-20:]:
            if not isinstance(t, dict):
                continue
            triggers_out.append(
                {
                    "layer": t.get("layer"),
                    "action": t.get("action"),
                    "triggered": t.get("triggered"),
                    "reason": t.get("reason"),
                }
            )
    pending = state.get("pending_memory_delta") or extra.get("_pending_memory_delta")
    pending_count = len(pending) if isinstance(pending, list) else 0
    out: dict[str, Any] = {
        "l0_applied": bool(state.get("l0_applied")),
        "pending_remember": bool(state.get("pending_remember") or extra.get("pending_remember")),
        "pending_memory_delta_count": pending_count,
        "layer_triggers": triggers_out,
    }
    if isinstance(turn_decision, dict):
        out["turn_decision"] = dict(turn_decision)
    return out


def _summarize_agent_events(events: Any) -> List[dict[str, Any]]:
    if not isinstance(events, list):
        return []
    out: List[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        out.append(
            {
                "type": ev.get("type"),
                "tool_name": ev.get("tool_name"),
                "duration_ms": ev.get("duration_ms"),
                "error": ev.get("error"),
            }
        )
    return out
