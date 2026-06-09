"""回忆预检索无结果时跳过 RAG。"""

from __future__ import annotations

from dataclasses import replace

from app.agents.chat_config import ChatAgentConfig
from app.agents.context_builder import SessionContextBuildResult
from app.agents.retrieval_router import RetrievalPlan


def _cfg(**kw):
    base = ChatAgentConfig(
        recall_skip_rag_when_prefetch_miss=True,
        retrieval_orchestration=True,
    )
    return replace(base, **kw) if kw else base


def test_recall_prefetch_miss_should_skip_rag_logic():
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
    assert run_rag is False
