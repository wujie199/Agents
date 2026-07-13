# -*- coding: utf-8 -*-
"""ConversationAuditMiddleware：turn 级审计包。"""

from __future__ import annotations

import json

import pytest

from app.agents.middleware.audit_content import apply_audit_content
from app.agents.middleware.conversation_audit import (
    ConversationAuditMiddleware,
    build_conversation_turn_entry,
)
from app.agents.middleware.privacy import PrivacyMiddleware


class _RunCtx:
    def __init__(self):
        self.request = type(
            "R",
            (),
            {
                "tenant_id": "t1",
                "user_id": "u1",
                "session_id": "s1",
                "trace_id": "tr-main",
            },
        )()
        self.extra = {
            "_active_turn_id": "turn-abc",
            "turn_audit": [{"node": "prepare", "duration_ms": 10.0, "error": None}],
            "layer_triggers": [
                {"layer": "L1", "action": "nudge", "triggered": True, "reason": "test"},
            ],
            "turn_decision": {"remember": False},
            "agent_events": [
                {"type": "tool_call", "tool_name": "rag_search", "duration_ms": 50},
            ],
        }


def test_apply_audit_content_redacted_masks_pii():
    raw = "联系我 13800138000 或 test@example.com"
    out = apply_audit_content(raw, "redacted")
    masked = out["text"]
    assert "13800138000" not in masked
    assert "test@example.com" not in masked
    assert PrivacyMiddleware.mask_pii(raw) == masked


def test_build_conversation_turn_entry_includes_prepare_and_memory():
    run_ctx = _RunCtx()
    prepare_stash = {
        "user_input": "扫地机器人怎么保养",
        "evidence_count": 2,
        "rag_empty": False,
        "memory_snapshot_hash": "abc123",
        "evidences_summary": [
            {
                "id": "doc1",
                "score": 0.9,
                "citation": "维护保养.txt",
                "content_preview": "定期清理尘盒",
            }
        ],
        "memory_summary": {"recall_hit": True, "retrieval_intent": "rag"},
        "retrieval_intent": "rag",
        "memory_path": "rag",
    }
    run_ctx.extra["_turn_audit_prepare"] = prepare_stash

    entry = build_conversation_turn_entry(
        state={"user_input": "扫地机器人怎么保养", "l0_applied": True},
        result={"assistant_text": "建议定期清理尘盒。"},
        run_ctx=run_ctx,
        ctx_extra={"trace_id": "tr-fallback"},
        error=None,
        content_mode="redacted",
        include_retrieval=True,
        include_tools=True,
        max_content_chars=8000,
    )

    assert entry["event_type"] == "conversation.turn"
    assert entry["turn_id"] == "turn-abc"
    assert entry["content_policy"] == "redacted"
    assert entry["user_input"]["policy"] == "redacted"
    assert "保养" in entry["user_input"]["text"]
    assert entry["prepare"]["evidence_count"] == 2
    assert entry["prepare"]["retrieval_intent"] == "rag"
    assert len(entry["prepare"]["evidences_summary"]) == 1
    assert entry["memory"]["l0_applied"] is True
    assert entry["memory"]["turn_decision"] == {"remember": False}
    assert entry["tools"][0]["tool_name"] == "rag_search"
    assert entry["nodes"][0]["node"] == "prepare"


@pytest.mark.asyncio
async def test_conversation_audit_prepare_stash_and_persist(tmp_path):
    run_ctx = _RunCtx()
    mw = ConversationAuditMiddleware(
        persist=True,
        audit_log_dir=str(tmp_path),
        audit_content="redacted",
    )
    config = {"configurable": {"run_ctx": run_ctx}}

    prepare_result = {
        "user_input": "手机号13800138000",
        "evidence_count": 1,
        "rag_empty": False,
        "evidences_summary": [{"id": "e1", "score": 0.8, "content_preview": "预览"}],
        "memory_summary": {"recall_hit": False},
        "retrieval_intent": "rag",
    }
    await mw.on_exit("prepare", {}, config, prepare_result)

    assert "_turn_audit_prepare" in run_ctx.extra
    assert run_ctx.extra["_turn_audit_prepare"]["evidence_count"] == 1

    persist_result = {"assistant_text": "好的，已记录。"}
    await mw.on_exit(
        "persist",
        {"user_input": "手机号13800138000", "l0_applied": False},
        config,
        persist_result,
        extra={"trace_id": "tr-persist"},
    )

    assert len(run_ctx.extra.get("conversation_turn_audit", [])) == 1
    files = list(tmp_path.glob("conversation_turn_*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert row["event_type"] == "conversation.turn"
    assert row["trace_id"] == "tr-main"
    assert "13800138000" not in row["user_input"]["text"]
    assert row["prepare"]["evidence_count"] == 1


@pytest.mark.asyncio
async def test_conversation_audit_skips_non_persist_nodes(tmp_path):
    run_ctx = _RunCtx()
    mw = ConversationAuditMiddleware(persist=True, audit_log_dir=str(tmp_path))
    config = {"configurable": {"run_ctx": run_ctx}}
    await mw.on_exit("agent", {}, config, {"assistant_text": "x"})
    assert list(tmp_path.glob("conversation_turn_*.jsonl")) == []
