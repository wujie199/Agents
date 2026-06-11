# -*- coding: utf-8 -*-
"""对话 Agent 配置（config/chat.yml）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import yaml


@dataclass(frozen=True)
class ChatAgentConfig:
    enable_rag: bool = True
    max_history_turns: int = 10
    max_evidence_chars: int = 6000
    max_evidence_items: int = 8
    rag_min_score: float = 0.0
    rag_min_keep: int = 3
    model_role: str = "main_llm"
    # Context Builder
    max_history_chars: int = 12000
    recent_turns: int = 6
    enable_rag_gating: bool = True
    rag_gating_min_chars: int = 4
    enable_rolling_summary: bool = True
    rolling_summary_max_chars: int = 1500
    use_llm_rolling_summary: bool = True
    summarizer_role: str = "memory_summarizer_llm"
    session_search_prefetch: bool = True
    session_search_prefetch_limit: int = 5
    session_search_prefetch_max_chars: int = 2000
    session_search_prefetch_scope: str = "auto"
    # 知识类 session_search 预检索：false=截断拼接（快、少幻觉）；回忆类见 session_search_llm_on_recall
    session_search_use_llm_summary: bool = False
    session_search_llm_on_recall: bool = True  # auto | session | user
    skill_prefetch: bool = True
    l4_profile_prefetch: bool = True
    # 检索编排（L0 路由）
    retrieval_orchestration: bool = True
    recall_wins_over_rag: bool = True
    knowledge_skips_recall: bool = True
    # 知识类问题是否同时预检索 L2 会话（与 RAG 并行，不互斥）
    knowledge_session_search: bool = True
    # 知识类问题是否注入 L4 外部画像
    l4_prefetch_on_knowledge: bool = True
    recall_skip_rag_when_prefetch_hit: bool = True
    recall_skip_rag_when_prefetch_miss: bool = False
    # 滚动摘要仅保留 user 轮次，避免 assistant 幻觉污染 context
    rolling_summary_user_only: bool = True
    # 有检索证据时强制 grounding，禁止编造未出现在证据中的事实
    evidence_strict_grounding: bool = True
    # 回忆预检索未命中时提示 Agent 调用 session_search 工具（Path B）
    recall_tool_hint_on_miss: bool = True
    retrieval_llm_router: bool = False
    retrieval_router_role: str = "router_llm"
    retrieval_router_confidence_min: float = 0.6
    # 知识/寒暄类直连 LLM（不走 ReAct 工具环，降低延迟）
    knowledge_direct_llm: bool = True
    # Path B 记忆工具
    enable_memory_tools: bool = True
    enable_skill_tools: bool = True
    enable_l4_tools: bool = True
    remember_require_hitl: bool = True
    # REPL 退出时自动确认剩余 pending L1（dev profile 默认 true）
    auto_confirm_pending_on_exit: bool = False
    # L2 交互写入
    interactive_flush_buffer: bool = True
    # finalize L2→L1
    enable_l1_extract_on_finalize: bool = True
    use_llm_l1_extract: bool = True
    l1_extract_max_turns: int = 20
    l1_extract_max_items: int = 5
    l1_extract_allowed_keys: Tuple[str, ...] = (
        "称呼",
        "姓名",
        "职业",
        "语言",
        "输出格式",
        "时区",
        "项目",
    )


_DEFAULT = ChatAgentConfig()


def _tuple_keys(raw: Any) -> Tuple[str, ...]:
    if not raw:
        return _DEFAULT.l1_extract_allowed_keys
    if isinstance(raw, (list, tuple)):
        return tuple(str(x) for x in raw)
    return _DEFAULT.l1_extract_allowed_keys


def _merge_chat_profile(chat: dict[str, Any], profile: str | None) -> dict[str, Any]:
    merged = dict(chat)
    profiles = merged.pop("profiles", None) or {}
    if profile and profile in profiles:
        overlay = profiles.get(profile) or {}
        if isinstance(overlay, dict):
            merged.update(overlay)
    return merged


def load_chat_config(
    config_dir: str | Path = "config",
    *,
    profile: str | None = None,
) -> ChatAgentConfig:
    path = Path(config_dir) / "chat.yml"
    if not path.is_file():
        return _DEFAULT
    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    chat = _merge_chat_profile(raw.get("chat") or {}, profile)
    return ChatAgentConfig(
        enable_rag=bool(chat.get("enable_rag", _DEFAULT.enable_rag)),
        max_history_turns=int(
            chat.get("max_history_turns", _DEFAULT.max_history_turns)
        ),
        max_evidence_chars=int(
            chat.get("max_evidence_chars", _DEFAULT.max_evidence_chars)
        ),
        max_evidence_items=int(
            chat.get("max_evidence_items", _DEFAULT.max_evidence_items)
        ),
        rag_min_score=float(chat.get("rag_min_score", _DEFAULT.rag_min_score)),
        rag_min_keep=int(chat.get("rag_min_keep", _DEFAULT.rag_min_keep)),
        model_role=str(chat.get("model_role", _DEFAULT.model_role)),
        max_history_chars=int(
            chat.get("max_history_chars", _DEFAULT.max_history_chars)
        ),
        recent_turns=int(chat.get("recent_turns", _DEFAULT.recent_turns)),
        enable_rag_gating=bool(
            chat.get("enable_rag_gating", _DEFAULT.enable_rag_gating)
        ),
        rag_gating_min_chars=int(
            chat.get("rag_gating_min_chars", _DEFAULT.rag_gating_min_chars)
        ),
        enable_rolling_summary=bool(
            chat.get("enable_rolling_summary", _DEFAULT.enable_rolling_summary)
        ),
        rolling_summary_max_chars=int(
            chat.get(
                "rolling_summary_max_chars", _DEFAULT.rolling_summary_max_chars
            )
        ),
        use_llm_rolling_summary=bool(
            chat.get("use_llm_rolling_summary", _DEFAULT.use_llm_rolling_summary)
        ),
        summarizer_role=str(
            chat.get("summarizer_role", _DEFAULT.summarizer_role)
        ),
        session_search_prefetch=bool(
            chat.get("session_search_prefetch", _DEFAULT.session_search_prefetch)
        ),
        session_search_prefetch_limit=int(
            chat.get(
                "session_search_prefetch_limit",
                _DEFAULT.session_search_prefetch_limit,
            )
        ),
        session_search_prefetch_max_chars=int(
            chat.get(
                "session_search_prefetch_max_chars",
                _DEFAULT.session_search_prefetch_max_chars,
            )
        ),
        session_search_prefetch_scope=str(
            chat.get(
                "session_search_prefetch_scope",
                _DEFAULT.session_search_prefetch_scope,
            )
        ),
        session_search_use_llm_summary=bool(
            chat.get(
                "session_search_use_llm_summary",
                _DEFAULT.session_search_use_llm_summary,
            )
        ),
        session_search_llm_on_recall=bool(
            chat.get(
                "session_search_llm_on_recall",
                _DEFAULT.session_search_llm_on_recall,
            )
        ),
        skill_prefetch=bool(chat.get("skill_prefetch", _DEFAULT.skill_prefetch)),
        l4_profile_prefetch=bool(
            chat.get("l4_profile_prefetch", _DEFAULT.l4_profile_prefetch)
        ),
        retrieval_orchestration=bool(
            chat.get("retrieval_orchestration", _DEFAULT.retrieval_orchestration)
        ),
        recall_wins_over_rag=bool(
            chat.get("recall_wins_over_rag", _DEFAULT.recall_wins_over_rag)
        ),
        knowledge_skips_recall=bool(
            chat.get("knowledge_skips_recall", _DEFAULT.knowledge_skips_recall)
        ),
        knowledge_session_search=bool(
            chat.get("knowledge_session_search", _DEFAULT.knowledge_session_search)
        ),
        l4_prefetch_on_knowledge=bool(
            chat.get("l4_prefetch_on_knowledge", _DEFAULT.l4_prefetch_on_knowledge)
        ),
        rolling_summary_user_only=bool(
            chat.get("rolling_summary_user_only", _DEFAULT.rolling_summary_user_only)
        ),
        evidence_strict_grounding=bool(
            chat.get("evidence_strict_grounding", _DEFAULT.evidence_strict_grounding)
        ),
        recall_tool_hint_on_miss=bool(
            chat.get("recall_tool_hint_on_miss", _DEFAULT.recall_tool_hint_on_miss)
        ),
        recall_skip_rag_when_prefetch_hit=bool(
            chat.get(
                "recall_skip_rag_when_prefetch_hit",
                _DEFAULT.recall_skip_rag_when_prefetch_hit,
            )
        ),
        recall_skip_rag_when_prefetch_miss=bool(
            chat.get(
                "recall_skip_rag_when_prefetch_miss",
                _DEFAULT.recall_skip_rag_when_prefetch_miss,
            )
        ),
        retrieval_llm_router=bool(
            chat.get("retrieval_llm_router", _DEFAULT.retrieval_llm_router)
        ),
        retrieval_router_role=str(
            chat.get("retrieval_router_role", _DEFAULT.retrieval_router_role)
        ),
        retrieval_router_confidence_min=float(
            chat.get(
                "retrieval_router_confidence_min",
                _DEFAULT.retrieval_router_confidence_min,
            )
        ),
        knowledge_direct_llm=bool(
            chat.get("knowledge_direct_llm", _DEFAULT.knowledge_direct_llm)
        ),
        enable_memory_tools=bool(
            chat.get("enable_memory_tools", _DEFAULT.enable_memory_tools)
        ),
        enable_skill_tools=bool(
            chat.get("enable_skill_tools", _DEFAULT.enable_skill_tools)
        ),
        enable_l4_tools=bool(
            chat.get("enable_l4_tools", _DEFAULT.enable_l4_tools)
        ),
        remember_require_hitl=bool(
            chat.get("remember_require_hitl", _DEFAULT.remember_require_hitl)
        ),
        auto_confirm_pending_on_exit=bool(
            chat.get(
                "auto_confirm_pending_on_exit",
                _DEFAULT.auto_confirm_pending_on_exit,
            )
        ),
        interactive_flush_buffer=bool(
            chat.get("interactive_flush_buffer", _DEFAULT.interactive_flush_buffer)
        ),
        enable_l1_extract_on_finalize=bool(
            chat.get(
                "enable_l1_extract_on_finalize",
                _DEFAULT.enable_l1_extract_on_finalize,
            )
        ),
        use_llm_l1_extract=bool(
            chat.get("use_llm_l1_extract", _DEFAULT.use_llm_l1_extract)
        ),
        l1_extract_max_turns=int(
            chat.get("l1_extract_max_turns", _DEFAULT.l1_extract_max_turns)
        ),
        l1_extract_max_items=int(
            chat.get("l1_extract_max_items", _DEFAULT.l1_extract_max_items)
        ),
        l1_extract_allowed_keys=_tuple_keys(chat.get("l1_extract_allowed_keys")),
    )
