# -*- coding: utf-8 -*-
"""检索编排：意图分类 + 通道互斥（L0 路由）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.agents.chat_config import ChatAgentConfig
from app.agents.context_builder import (
    is_l4_query,
    is_recall_query,
    is_skill_query,
    should_run_rag,
)

Intent = str  # recall | knowledge | skill | profile | chitchat | legacy


@dataclass(frozen=True)
class RetrievalPlan:
    """单轮检索通道计划。"""

    intent: Intent
    run_session_search: bool
    run_rag: bool
    run_skill: bool
    run_l4: bool
    skip_rag_reason: Optional[str] = None
    skip_recall_reason: Optional[str] = None

    def to_debug_dict(self) -> dict[str, Any]:
        channels = []
        if self.run_session_search:
            channels.append("SESSION_SEARCH")
        if self.run_rag:
            channels.append("RAG")
        if self.run_skill:
            channels.append("SKILL")
        if self.run_l4:
            channels.append("L4")
        skipped: dict[str, str] = {}
        if self.skip_rag_reason:
            skipped["RAG"] = self.skip_rag_reason
        if self.skip_recall_reason:
            skipped["SESSION_SEARCH"] = self.skip_recall_reason
        return {
            "intent": self.intent,
            "channels": channels,
            "skipped": skipped,
        }


def should_use_direct_llm_for_intent(intent: Intent, cfg: ChatAgentConfig) -> bool:
    """知识/寒暄在证据已预注入时可直连 LLM，回忆/profile/skill 仍走 ReAct。"""
    if not cfg.enable_memory_tools:
        return True
    if not cfg.knowledge_direct_llm:
        return False
    return intent in ("knowledge", "chitchat")


def _plan_from_intent(
    intent: Intent,
    query: str,
    cfg: ChatAgentConfig,
    *,
    enable_rag: bool,
) -> RetrievalPlan:
    run_session_search = False
    run_rag = False
    run_skill = False
    run_l4 = False
    skip_rag: Optional[str] = None
    skip_recall: Optional[str] = None

    if intent == "recall" and cfg.session_search_prefetch:
        run_session_search = True
    elif intent == "knowledge" and enable_rag and cfg.enable_rag:
        run_rag = should_run_rag(query, cfg)
        if (
            cfg.knowledge_session_search
            and cfg.session_search_prefetch
        ):
            run_session_search = True
        if (
            cfg.l4_prefetch_on_knowledge
            and cfg.l4_profile_prefetch
            and cfg.enable_l4_tools
            and is_l4_query(query)
        ):
            run_l4 = True
    elif intent == "skill" and cfg.skill_prefetch and cfg.enable_skill_tools:
        run_skill = True
    elif intent == "profile":
        if cfg.l4_profile_prefetch and cfg.enable_l4_tools:
            run_l4 = True
        skip_rag = "profile_intent"
        skip_recall = "profile_intent"

    if cfg.recall_wins_over_rag and intent == "recall" and run_rag:
        run_rag = False
        skip_rag = "recall_intent"
    elif cfg.recall_wins_over_rag and intent == "recall":
        skip_rag = "recall_intent"

    if (
        cfg.knowledge_skips_recall
        and not cfg.knowledge_session_search
        and intent == "knowledge"
        and run_session_search
    ):
        run_session_search = False
        skip_recall = "knowledge_intent"
    elif (
        cfg.knowledge_skips_recall
        and not cfg.knowledge_session_search
        and intent == "knowledge"
    ):
        skip_recall = "knowledge_intent"

    return RetrievalPlan(
        intent=intent,
        run_session_search=run_session_search,
        run_rag=run_rag,
        run_skill=run_skill,
        run_l4=run_l4,
        skip_rag_reason=skip_rag,
        skip_recall_reason=skip_recall,
    )


def classify_intent(query: str, cfg: ChatAgentConfig) -> Intent:
    q = (query or "").strip()
    if is_skill_query(q):
        return "skill"
    if is_l4_query(q):
        return "profile"
    if is_recall_query(q):
        return "recall"
    if not should_run_rag(q, cfg):
        return "chitchat"
    return "knowledge"


def _legacy_plan(
    query: str,
    cfg: ChatAgentConfig,
    *,
    enable_rag: bool,
) -> RetrievalPlan:
    """关闭 orchestration 时保持原有独立门控行为。"""
    return RetrievalPlan(
        intent="legacy",
        run_session_search=bool(
            cfg.session_search_prefetch and is_recall_query(query)
        ),
        run_rag=bool(enable_rag and cfg.enable_rag and should_run_rag(query, cfg)),
        run_skill=bool(
            cfg.skill_prefetch
            and cfg.enable_skill_tools
            and is_skill_query(query)
        ),
        run_l4=bool(
            cfg.l4_profile_prefetch
            and cfg.enable_l4_tools
            and is_l4_query(query)
        ),
    )


def build_retrieval_plan(
    query: str,
    cfg: ChatAgentConfig,
    *,
    enable_rag: bool = True,
) -> RetrievalPlan:
    if not cfg.retrieval_orchestration:
        return _legacy_plan(query, cfg, enable_rag=enable_rag)

    intent = classify_intent(query, cfg)
    return _plan_from_intent(intent, query, cfg, enable_rag=enable_rag)


def _log_retrieval_plan(
    ctx: Any,
    query: str,
    plan: RetrievalPlan,
    *,
    intent_source: str = "regex",
) -> None:
    try:
        from app.agents.memory_runtime_debug import trace_layer_trigger

        trace_layer_trigger(
            ctx,
            "ROUTER",
            "build_plan",
            True,
            intent_source,
            data={
                "intent": plan.intent,
                "query_preview": (query or "")[:80],
                **plan.to_debug_dict(),
            },
        )
    except Exception:
        pass


async def build_retrieval_plan_async(
    query: str,
    cfg: ChatAgentConfig,
    *,
    enable_rag: bool = True,
    models: Any = None,
    ctx: Any = None,
) -> RetrievalPlan:
    if not cfg.retrieval_orchestration:
        plan = _legacy_plan(query, cfg, enable_rag=enable_rag)
        _log_retrieval_plan(ctx, query, plan, intent_source="legacy")
        return plan

    intent = classify_intent(query, cfg)
    intent_source = "regex"
    if cfg.retrieval_llm_router and models is not None:
        from app.agents.retrieval_llm import classify_intent_llm

        llm_intent, confidence = await classify_intent_llm(query, cfg, models)
        if confidence >= cfg.retrieval_router_confidence_min:
            intent = llm_intent
            intent_source = f"llm:{confidence:.2f}"
    plan = _plan_from_intent(intent, query, cfg, enable_rag=enable_rag)
    _log_retrieval_plan(ctx, query, plan, intent_source=intent_source)
    return plan
