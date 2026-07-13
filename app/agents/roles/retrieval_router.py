# -*- coding: utf-8 -*-
"""检索编排：意图分类 + 通道互斥（L0 路由）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.context_builder import (
    is_knowledge_like,
    is_l4_query,
    is_meta_recall_query,
    is_mixed_recall_knowledge,
    is_name_intro_query,
    is_recall_query,
    is_skill_query,
    resolve_recall_scope,
    should_run_rag,
)

Intent = str  # recall | knowledge | skill | profile | chitchat | recall_and_knowledge | legacy
RecallStrategy = str  # none | meta_recent | semantic | browse
ROUTING_RULES_VERSION = "2026.07.06-p0"


@dataclass(frozen=True)
class RetrievalPlan:
    """单轮检索通道计划。"""

    intent: Intent
    run_session_search: bool
    run_rag: bool
    run_skill: bool
    run_l4: bool
    recall_strategy: RecallStrategy = "none"
    recall_scope: str = "session"
    intent_source: str = "regex"
    rules_version: str = ROUTING_RULES_VERSION
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
            "recall_strategy": self.recall_strategy,
            "recall_scope": self.recall_scope,
            "intent_source": self.intent_source,
            "rules_version": self.rules_version,
        }

    def to_audit_dict(self) -> dict[str, Any]:
        """Turn 审计 / metrics 用精简路由包。"""
        return {
            "intent": self.intent,
            "recall_strategy": self.recall_strategy,
            "recall_scope": self.recall_scope,
            "intent_source": self.intent_source,
            "rules_version": self.rules_version,
            "run_session_search": self.run_session_search,
            "run_rag": self.run_rag,
            "run_skill": self.run_skill,
            "run_l4": self.run_l4,
            "skip_rag_reason": self.skip_rag_reason,
            "skip_recall_reason": self.skip_recall_reason,
        }


def resolve_recall_strategy(
    query: str,
    cfg: ChatAgentConfig,
    *,
    intent: Intent,
) -> RecallStrategy:
    """L2 预检索策略：meta 枚举 / 跨会话 browse / 语义 discovery。"""
    if intent in ("skill", "profile", "chitchat"):
        return "none"
    if intent == "recall" or (intent == "legacy" and is_recall_query(query)):
        if cfg.meta_recall_recent_list and is_meta_recall_query(query):
            if resolve_recall_scope(query, cfg) == "user":
                return "browse"
            return "meta_recent"
        return "semantic"
    if intent in ("knowledge", "recall_and_knowledge", "legacy"):
        return "semantic"
    return "none"


def _attach_recall_routing(
    plan: RetrievalPlan,
    query: str,
    cfg: ChatAgentConfig,
) -> RetrievalPlan:
    if not plan.run_session_search:
        return plan
    scope = resolve_recall_scope(query, cfg)
    strategy = resolve_recall_strategy(query, cfg, intent=plan.intent)
    return RetrievalPlan(
        intent=plan.intent,
        run_session_search=plan.run_session_search,
        run_rag=plan.run_rag,
        run_skill=plan.run_skill,
        run_l4=plan.run_l4,
        recall_strategy=strategy,
        recall_scope=scope,
        intent_source=plan.intent_source,
        rules_version=plan.rules_version,
        skip_rag_reason=plan.skip_rag_reason,
        skip_recall_reason=plan.skip_recall_reason,
    )


def classify_intent_hard_rules(query: str, cfg: ChatAgentConfig) -> Optional[Intent]:
    """硬规则 fast path：skill / profile / 寒暄，LLM 不可覆盖。"""
    q = (query or "").strip()
    if not q:
        return "chitchat"
    if is_skill_query(q):
        return "skill"
    if is_l4_query(q) or is_name_intro_query(q):
        return "profile"
    if not should_run_rag(q, cfg):
        return "chitchat"
    return None


def is_ambiguous_for_llm_router(query: str, cfg: ChatAgentConfig) -> bool:
    """仅歧义 query 才调用 LLM router（硬规则与清晰 regex 跳过）。"""
    q = (query or "").strip()
    if not q or classify_intent_hard_rules(q, cfg) is not None:
        return False
    if is_meta_recall_query(q):
        return False
    if is_recall_query(q) and is_knowledge_like(q) and not is_meta_recall_query(q):
        return True
    if is_recall_query(q) and not is_meta_recall_query(q):
        if re.search(r"(上次|之前|刚才).+(说|问|提|讨论|聊).+", q):
            return False
        return True
    if len(q) >= 14 and not is_recall_query(q):
        return False
    return len(q) < 10


def validate_llm_intent(query: str, cfg: ChatAgentConfig, llm_intent: Intent) -> bool:
    """LLM 意图 guardrail：不得违背硬规则信号。"""
    q = (query or "").strip()
    allowed = {
        "recall",
        "knowledge",
        "skill",
        "profile",
        "chitchat",
        "recall_and_knowledge",
    }
    if llm_intent not in allowed:
        return False
    if llm_intent == "skill" and not is_skill_query(q):
        return False
    if llm_intent == "profile" and not (is_l4_query(q) or is_name_intro_query(q)):
        return False
    if llm_intent == "chitchat" and should_run_rag(q, cfg):
        return False
    hard = classify_intent_hard_rules(q, cfg)
    if hard is not None and llm_intent != hard:
        return False
    return True


def resolve_intent(query: str, cfg: ChatAgentConfig) -> tuple[Intent, str]:
    """同步意图解析：硬规则 → 正则。"""
    hard = classify_intent_hard_rules(query, cfg)
    if hard is not None:
        return hard, "hard_rule"
    return classify_intent(query, cfg), "regex"


async def resolve_intent_async(
    query: str,
    cfg: ChatAgentConfig,
    *,
    models: Any = None,
) -> tuple[Intent, str]:
    """异步意图解析：硬规则 → 正则 →（歧义时）LLM guardrail。"""
    intent, source = resolve_intent(query, cfg)
    if source == "hard_rule":
        return intent, source
    if not cfg.retrieval_llm_router or models is None:
        return intent, source
    if not is_ambiguous_for_llm_router(query, cfg):
        return intent, source
    from app.agents.prompts.retrieval_llm import classify_intent_llm

    llm_intent, confidence = await classify_intent_llm(query, cfg, models)
    if confidence < cfg.retrieval_router_confidence_min:
        return intent, source
    if not validate_llm_intent(query, cfg, llm_intent):
        return intent, f"regex:llm_rejected:{confidence:.2f}"
    return llm_intent, f"llm:{confidence:.2f}"


def _plan_with_routing_meta(
    plan: RetrievalPlan,
    *,
    intent_source: str,
) -> RetrievalPlan:
    return RetrievalPlan(
        intent=plan.intent,
        run_session_search=plan.run_session_search,
        run_rag=plan.run_rag,
        run_skill=plan.run_skill,
        run_l4=plan.run_l4,
        recall_strategy=plan.recall_strategy,
        recall_scope=plan.recall_scope,
        intent_source=intent_source,
        rules_version=ROUTING_RULES_VERSION,
        skip_rag_reason=plan.skip_rag_reason,
        skip_recall_reason=plan.skip_recall_reason,
    )


def should_use_direct_llm_for_intent(intent: Intent, cfg: ChatAgentConfig) -> bool:
    """知识/寒暄/混合意图在证据已预注入时可直连 LLM，回忆/profile/skill 仍走 ReAct。"""
    if not cfg.enable_memory_tools:
        return True
    if not cfg.knowledge_direct_llm:
        return False
    return intent in ("knowledge", "chitchat", "recall_and_knowledge")


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
    elif intent == "recall_and_knowledge" and cfg.recall_rag_fusion:
        # 混合意图：回忆 + RAG 并行
        run_session_search = cfg.session_search_prefetch
        run_rag = enable_rag and cfg.enable_rag and should_run_rag(query, cfg)
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
    elif intent == "skill":
        if cfg.skill_prefetch and cfg.enable_skill_tools:
            run_skill = True
        skip_rag = "skill_intent"
        skip_recall = "skill_intent"
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


def _finalize_plan(query: str, cfg: ChatAgentConfig, plan: RetrievalPlan) -> RetrievalPlan:
    return _attach_recall_routing(plan, query, cfg)


def classify_intent(query: str, cfg: ChatAgentConfig) -> Intent:
    q = (query or "").strip()
    hard = classify_intent_hard_rules(q, cfg)
    if hard is not None:
        return hard
    if is_meta_recall_query(q):
        return "recall"
    # 混合意图：回忆 + 求结论/对比（如"上次问的那个方案怎么样了"）
    if (
        cfg.recall_rag_fusion
        and is_mixed_recall_knowledge(q)
        and not is_meta_recall_query(q)
    ):
        return "recall_and_knowledge"
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
        base = _legacy_plan(query, cfg, enable_rag=enable_rag)
        _, source = resolve_intent(query, cfg)
        return _finalize_plan(
            query,
            cfg,
            _plan_with_routing_meta(base, intent_source=source),
        )

    intent, source = resolve_intent(query, cfg)
    base = _plan_from_intent(intent, query, cfg, enable_rag=enable_rag)
    return _finalize_plan(query, cfg, _plan_with_routing_meta(base, intent_source=source))


def _log_retrieval_plan(
    ctx: Any,
    query: str,
    plan: RetrievalPlan,
    *,
    intent_source: str = "regex",
) -> None:
    try:
        from app.agents.memory.memory_runtime_debug import trace_layer_trigger

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
        intent, source = resolve_intent(query, cfg)
        plan = _finalize_plan(
            query,
            cfg,
            _plan_with_routing_meta(
                _legacy_plan(query, cfg, enable_rag=enable_rag),
                intent_source=source,
            ),
        )
        _log_retrieval_plan(ctx, query, plan, intent_source=plan.intent_source)
        return plan

    intent, source = await resolve_intent_async(query, cfg, models=models)
    base = _plan_from_intent(intent, query, cfg, enable_rag=enable_rag)
    plan = _finalize_plan(query, cfg, _plan_with_routing_meta(base, intent_source=source))
    _log_retrieval_plan(ctx, query, plan, intent_source=plan.intent_source)
    if ctx is not None and isinstance(getattr(ctx, "extra", None), dict):
        ctx.extra["retrieval_plan_audit"] = plan.to_audit_dict()
    return plan
