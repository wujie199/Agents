# -*- coding: utf-8 -*-
"""对话 Agent 配置（config/chat.yml）。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool = True
    trace_header: str = "X-Request-ID"
    slow_threshold_ms: Dict[str, int] = field(
        default_factory=lambda: {
            "prepare": 2000,
            "agent": 8000,
            "persist": 500,
        }
    )
    rate_limit_backend: str = "memory"
    max_qps_per_tenant: int = 100
    audit_persist: bool = False
    audit_log_dir: str = "data/audit"
    audit_content: str = "redacted"
    audit_include_retrieval: bool = True
    audit_include_tools: bool = True
    audit_max_content_chars: int = 8000


_DEFAULT_OBS = ObservabilityConfig()


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
    # meta 回忆（问过什么/聊过什么）：按时间列最近 user 发言，非语义检索
    meta_recall_recent_list: bool = True
    meta_recall_recent_limit: int = 10
    skill_prefetch: bool = True
    l4_profile_prefetch: bool = True
    # 检索编排（L0 路由）
    retrieval_orchestration: bool = True
    recall_wins_over_rag: bool = True
    knowledge_skips_recall: bool = True
    # 回忆 + RAG 并行融合（打破严格互斥）
    recall_rag_fusion: bool = True
    fusion_recall_weight: float = 0.6
    fusion_rag_weight: float = 0.4
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
    # 知识类问题 RAG 未命中时禁止幻觉作答
    refuse_hallucination_on_rag_miss: bool = True
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
    # DeepAgents 规划层（可选，需 pip install 'agents[planning]'）
    enable_deep_agent: bool = False
    deep_agent_semantic_gate: bool = True
    deep_agent_gate_threshold: float = 0.7
    # L1 置信度分层自动写入
    l1_auto_write_confidence_min: float = 0.9
    l1_conflict_strategy: str = "ask_user"  # overwrite | keep_old | ask_user
    # 时间衰减
    time_decay: bool = True
    time_decay_half_life_days: float = 90.0
    # L0 上下文压缩（Hermes Phase A）
    l0_context_compress_enabled: bool = True
    context_compress_threshold: float = 0.50
    compress_target_ratio: float = 0.20
    context_window_tokens: int = 128000


_DEFAULT = ChatAgentConfig()


def _deep_merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """chat.yml 基座 + 增量：嵌套 dict 递归合并，标量/列表以 overlay 为准。"""
    result = dict(base)
    for key, overlay_val in overlay.items():
        base_val = result.get(key)
        if isinstance(overlay_val, dict) and isinstance(base_val, dict):
            result[key] = _deep_merge_config(base_val, overlay_val)
        else:
            result[key] = overlay_val
    return result


def chat_base_path(config_dir: str | Path = "config") -> Path:
    return Path(config_dir) / "chat.yml"


def resolve_chat_config_path(config_dir: str | Path = "config") -> Path:
    """CHAT_CONFIG 环境变量 > config/chat.yml。"""
    env_path = os.environ.get("CHAT_CONFIG")
    if env_path:
        return Path(env_path)
    return chat_base_path(config_dir)


def load_chat_yaml_document(
    config_path: str | Path,
    *,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """加载 Chat YAML；非 chat.yml 与基座深合并。"""
    path = Path(config_path)
    raw: dict[str, Any]
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base_path = chat_base_path(config_dir)
    if path.resolve() != base_path.resolve():
        base_raw: dict[str, Any] = {}
        if base_path.is_file():
            with base_path.open(encoding="utf-8") as f:
                base_raw = yaml.safe_load(f) or {}
        if base_raw:
            return _deep_merge_config(base_raw, raw)
    return raw


def _load_concurrency_default(config_dir: str | Path = "config") -> dict[str, Any]:
    path = Path(config_dir) / "concurrency.yml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    block = raw.get("default")
    return dict(block) if isinstance(block, dict) else {}


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


def load_observability_config(
    config_dir: str | Path = "config",
    *,
    profile: str | None = None,
) -> ObservabilityConfig:
    path = resolve_chat_config_path(config_dir)
    if not path.is_file():
        return _DEFAULT_OBS
    raw = load_chat_yaml_document(path, config_dir=config_dir)
    obs = raw.get("observability") or {}
    conc = _load_concurrency_default(config_dir)
    thresholds = dict(_DEFAULT_OBS.slow_threshold_ms)
    raw_thresholds = obs.get("slow_threshold_ms")
    if isinstance(raw_thresholds, dict):
        for key, value in raw_thresholds.items():
            thresholds[str(key)] = int(value)
    backend = str(obs.get("rate_limit_backend", _DEFAULT_OBS.rate_limit_backend))
    if obs.get("rate_limit_backend") == "redis" or os.environ.get("REDIS_URL"):
        backend = "redis"
    default_qps = int(conc.get("max_qps_per_tenant", _DEFAULT_OBS.max_qps_per_tenant))
    return ObservabilityConfig(
        enabled=bool(obs.get("enabled", _DEFAULT_OBS.enabled)),
        trace_header=str(obs.get("trace_header", _DEFAULT_OBS.trace_header)),
        slow_threshold_ms=thresholds,
        rate_limit_backend=backend,
        max_qps_per_tenant=int(obs.get("max_qps_per_tenant", default_qps)),
        audit_persist=bool(obs.get("audit_persist", _DEFAULT_OBS.audit_persist)),
        audit_log_dir=str(obs.get("audit_log_dir", _DEFAULT_OBS.audit_log_dir)),
        audit_content=str(obs.get("audit_content", _DEFAULT_OBS.audit_content)),
        audit_include_retrieval=bool(
            obs.get("audit_include_retrieval", _DEFAULT_OBS.audit_include_retrieval)
        ),
        audit_include_tools=bool(
            obs.get("audit_include_tools", _DEFAULT_OBS.audit_include_tools)
        ),
        audit_max_content_chars=int(
            obs.get("audit_max_content_chars", _DEFAULT_OBS.audit_max_content_chars)
        ),
    )


def load_chat_config(
    config_dir: str | Path = "config",
    *,
    profile: str | None = None,
) -> ChatAgentConfig:
    path = resolve_chat_config_path(config_dir)
    if not path.is_file():
        return _DEFAULT
    raw = load_chat_yaml_document(path, config_dir=config_dir)
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
        meta_recall_recent_list=bool(
            chat.get("meta_recall_recent_list", _DEFAULT.meta_recall_recent_list)
        ),
        meta_recall_recent_limit=int(
            chat.get("meta_recall_recent_limit", _DEFAULT.meta_recall_recent_limit)
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
        recall_rag_fusion=bool(
            chat.get("recall_rag_fusion", _DEFAULT.recall_rag_fusion)
        ),
        fusion_recall_weight=float(
            chat.get("fusion_recall_weight", _DEFAULT.fusion_recall_weight)
        ),
        fusion_rag_weight=float(
            chat.get("fusion_rag_weight", _DEFAULT.fusion_rag_weight)
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
        refuse_hallucination_on_rag_miss=bool(
            chat.get(
                "refuse_hallucination_on_rag_miss",
                _DEFAULT.refuse_hallucination_on_rag_miss,
            )
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
        enable_deep_agent=bool(
            chat.get("enable_deep_agent", _DEFAULT.enable_deep_agent)
        ),
        deep_agent_semantic_gate=bool(
            chat.get("deep_agent_semantic_gate", _DEFAULT.deep_agent_semantic_gate)
        ),
        deep_agent_gate_threshold=float(
            chat.get("deep_agent_gate_threshold", _DEFAULT.deep_agent_gate_threshold)
        ),
        l1_auto_write_confidence_min=float(
            chat.get("l1_auto_write_confidence_min", _DEFAULT.l1_auto_write_confidence_min)
        ),
        l1_conflict_strategy=str(
            chat.get("l1_conflict_strategy", _DEFAULT.l1_conflict_strategy)
        ),
        time_decay=bool(
            chat.get("time_decay", _DEFAULT.time_decay)
        ),
        time_decay_half_life_days=float(
            chat.get("time_decay_half_life_days", _DEFAULT.time_decay_half_life_days)
        ),
        l0_context_compress_enabled=bool(
            chat.get("l0_context_compress_enabled", _DEFAULT.l0_context_compress_enabled)
        ),
        context_compress_threshold=float(
            chat.get(
                "context_compress_threshold",
                _DEFAULT.context_compress_threshold,
            )
        ),
        compress_target_ratio=float(
            chat.get("compress_target_ratio", _DEFAULT.compress_target_ratio)
        ),
        context_window_tokens=int(
            chat.get("context_window_tokens", _DEFAULT.context_window_tokens)
        ),
    )
