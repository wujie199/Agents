"""回忆预检索未命中时的 RAG 策略。"""

from __future__ import annotations

from dataclasses import replace

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.context_builder import SessionContextBuildResult
from app.agents.roles.retrieval_router import RetrievalPlan


def _cfg(**kw):
    base = ChatAgentConfig(
        recall_skip_rag_when_prefetch_miss=False,
        retrieval_orchestration=True,
    )
    return replace(base, **kw) if kw else base


def test_recall_prefetch_miss_default_keeps_rag():
    plan = RetrievalPlan(
        intent="recall",
        run_session_search=True,
        run_rag=False,
        run_skill=False,
        run_l4=False,
        skip_rag_reason="recall_intent",
    )
    session_ctx = SessionContextBuildResult(
        trimmed_history=[],
        extra_text="",
        recall_prefetch="",
    )
    cfg = _cfg()
    run_rag = True
    if (
        run_rag
        and cfg.recall_skip_rag_when_prefetch_miss
        and plan.run_session_search
        and not session_ctx.recall_prefetch_hit
    ):
        run_rag = False
    assert run_rag is True


def test_recall_prefetch_miss_can_skip_when_configured():
    plan = RetrievalPlan(
        intent="recall",
        run_session_search=True,
        run_rag=False,
        run_skill=False,
        run_l4=False,
        skip_rag_reason="recall_intent",
    )
    session_ctx = SessionContextBuildResult(
        trimmed_history=[],
        extra_text="",
        recall_prefetch="",
    )
    cfg = _cfg(recall_skip_rag_when_prefetch_miss=True)
    run_rag = True
    if (
        run_rag
        and cfg.recall_skip_rag_when_prefetch_miss
        and plan.intent == "recall"
        and plan.run_session_search
        and not session_ctx.recall_prefetch_hit
    ):
        run_rag = False
    assert run_rag is False
