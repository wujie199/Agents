# -*- coding: utf-8 -*-
"""企业级 Context 组装：RAG 门控、历史字符预算、滚动摘要、回忆预检索。"""

from __future__ import annotations

import json
import re
import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from core.composition.run_context import RunContext

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.debug.debug_trace import agent_debug
from app.agents.prompts.text_sanitize import sanitize_turn_content, strip_model_reasoning

if TYPE_CHECKING:
    from app.agents.roles.retrieval_router import RetrievalPlan

# 寒暄 / 元对话：跳过 RAG
_RAG_SKIP_RE = re.compile(
    r"^(你好|您好|hi|hello|hey|在吗|谢谢|感谢|再见|拜拜|"
    r"你叫什么|你是谁|什么模型|who are you|"
    r"好的|ok|okay|嗯|对|是的|没错|不行|不用|不要|"
    r"继续|停|退出|结束|取消)[\s?!.，。~]*$",
    re.I,
)

# 回忆类：可预检索 session_search
_RECALL_RE = re.compile(
    r"之前|上次|刚才|还记得|记得吗|说过什么|问过什么|我们聊|早些时候|前面|"
    r"之前聊|之前问|上次说的|刚才提到的|前面提到的|之前提到的|还记得吗|"
    r"都聊|聊过|历史问题",
)

# 枚举型 meta 回忆（列最近 user 发言，不做语义 session_search）
_META_RECALL_RE = re.compile(
    r"(?:问过|说过|聊过|讨论过)(?:什么|哪些|啥)|"
    r"(?:什么|哪些)问题|有哪些问题|历史问题|都(?:问|聊|说)过什么|记得(?:我)?问|"
    r"聊(?:过|了)什么|我们聊(?:过|了)?(?:什么|啥)|"
    r"之前(?:都)?(?:问|聊|说)了?什么",
)

# 知识倾向：query 中同时包含回忆 + 知识检索关键词（混合意图）
_KNOWLEDGE_LIKE_RE = re.compile(
    r"怎么样|如何|什么|哪个|区别|对比|分析|原理|机制|定义|方案|建议|"
    r"推荐|好不好|行不行|能不能|是不是|怎么选",
)

# 跨会话回忆
_CROSS_SESSION_RE = re.compile(
    r"别的会话|其他会话|历史会话|上次会话|另一个会话|跨会话|所有会话",
)

# 技能意图（L3：显式技能/流程/模板）
_SKILL_RE = re.compile(
    r"技能|skill|"
    r"用.{0,12}技能|按.{0,12}技能|运行.{0,12}技能|执行.{0,12}技能|"
    r"用.*工具|执行.*流程|跑一下.*流程|"
    r"json_section|report_context|session_lookup|list_json_titles|"
    r"模板|工作流|workflow",
    re.I,
)

# L4 画像 / 实体 / 用户身份（走 profile intent，跳过 RAG 与 session_search）
_L4_RE = re.compile(
    r"部门|职位|工号|CRM|画像|我是谁|我的资料|实体|别名|LDAP|"
    r"叫什么名字|叫什么名|我的名字|我的姓名|姓名是什么|"
    r"你知道我的|我的哪些信息|关于我什么|个人资料|我的信息|"
    r"我是哪个|我是什么部门|我是什么职位",
    re.I,
)
_NAME_INTRO_RE = re.compile(
    r"^(我叫|我的名字是|名字是|我是)",
    re.I,
)
_NAME_SHORT_RE = re.compile(r"^叫[\u4e00-\u9fa5a-zA-Z]{1,12}$")

_EMPTY_RECALL_MARKERS = (
    "no relevant messages found",
    "archive search not available",
    "search error:",
    "未找到相关",
)


def is_empty_session_search_text(text: str) -> bool:
    """session_search 无有效命中时的占位文案。"""
    t = (text or "").strip()
    if not t:
        return True
    lower = t.lower()
    return any(m in lower for m in _EMPTY_RECALL_MARKERS)


@dataclass(frozen=True)
class SessionContextBuildResult:
    trimmed_history: List[dict]
    extra_text: str
    recall_prefetch: str = ""

    @property
    def recall_prefetch_hit(self) -> bool:
        text = (self.recall_prefetch or "").strip()
        if not text:
            return False
        lower = text.lower()
        return not any(m in lower for m in _EMPTY_RECALL_MARKERS)


def should_run_rag(query: str, cfg: ChatAgentConfig) -> bool:
    if not cfg.enable_rag_gating:
        return True
    q = (query or "").strip()
    if len(q) < cfg.rag_gating_min_chars:
        return False
    if _RAG_SKIP_RE.match(q):
        return False
    return True


def is_recall_query(query: str) -> bool:
    return bool(_RECALL_RE.search((query or "").strip()))


_MIXED_EVAL_RE = re.compile(
    r"怎么样|如何|好不好|行不行|能不能|是不是|对不对|多少|什么意思|哪个更好|能行吗|行吗",
)


def is_meta_recall_query(query: str) -> bool:
    """枚举型回忆：列最近 user 问题，而非语义检索特定话题。"""
    q = (query or "").strip()
    if not q:
        return False
    # 「什么品牌 / 什么方案」→ 指向具体话题，非 meta
    m = re.search(r"什么(.+)$", q)
    if m:
        rest = re.sub(r"[\s?!.，。~？！吗呢]+$", "", m.group(1)).strip()
        if rest and rest not in ("问题", "内容"):
            return False
    if re.search(r"(是什么|是啥|是哪些)$", q):
        return False
    if re.search(
        r"(问过什么|说过什么|聊过什么|问过哪些|说过哪些|聊过哪些|"
        r"什么问题|哪些内容|有哪些问题|聊过什么内容)[\s?!.，。~？！吗呢]*$",
        q,
    ):
        return True
    if _META_RECALL_RE.search(q):
        return True
    if not is_recall_query(q):
        return False
    # 「上次/之前 + 具体话题 + 怎么样」→ 指向性回忆
    if is_knowledge_like(q) and re.search(
        r"(上次|之前|刚才).+(说|问|提|讨论|聊)", q
    ):
        return False
    return False


def is_mixed_recall_knowledge(query: str) -> bool:
    """混合意图：回忆 + 明确求结论/对比（非 meta 枚举）。"""
    q = (query or "").strip()
    if is_meta_recall_query(q):
        return False
    if not is_recall_query(q):
        return False
    return bool(_MIXED_EVAL_RE.search(q))


def is_cross_session_recall(query: str) -> bool:
    return bool(_CROSS_SESSION_RE.search((query or "").strip()))


def resolve_recall_scope(query: str, cfg: ChatAgentConfig) -> str:
    """返回 session_search scope：session 或 user。"""
    explicit = (cfg.session_search_prefetch_scope or "auto").lower()
    if explicit in ("session", "user"):
        return explicit
    if is_cross_session_recall(query):
        return "user"
    return "session"


def is_skill_query(query: str) -> bool:
    return bool(_SKILL_RE.search((query or "").strip()))


def is_knowledge_like(query: str) -> bool:
    """query 包含知识倾向关键词（如"怎么样"、"如何"、"区别"等）。"""
    return bool(_KNOWLEDGE_LIKE_RE.search((query or "").strip()))


def is_l4_query(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if is_name_intro_query(q):
        return True
    return bool(_L4_RE.search(q))


def is_name_intro_query(query: str) -> bool:
    """用户自述/更正姓名（过短时不应落 chitchat）。"""
    q = (query or "").strip()
    if not q:
        return False
    if _NAME_INTRO_RE.match(q):
        return True
    return bool(_NAME_SHORT_RE.match(q))


def dedupe_history_turns(
    history: List[dict],
    *,
    collapse_duplicate_user: bool = False,
) -> List[dict]:
    """去掉连续重复轮次；可选折叠相同 user 内容（保留最近一次）。"""
    if not history:
        return []
    deduped: List[dict] = []
    prev_key: Optional[tuple[str, str]] = None
    for row in history:
        role = str(row.get("role") or "")
        content = (row.get("content") or "").strip()
        if not content:
            continue
        key = (role, content)
        if key == prev_key:
            continue
        deduped.append(row)
        prev_key = key
    if not collapse_duplicate_user:
        return deduped
    seen_user: set[str] = set()
    collapsed: List[dict] = []
    for row in reversed(deduped):
        if row.get("role") == "user":
            content = (row.get("content") or "").strip()
            if content in seen_user:
                continue
            seen_user.add(content)
        collapsed.append(row)
    collapsed.reverse()
    return collapsed


def compress_history_for_knowledge(
    history: List[dict],
    *,
    max_assistant_chars: int = 180,
    keep_assistant_turns: int = 2,
) -> List[dict]:
    """知识类：保留 user 原文，assistant 仅最近 N 条且截断。"""
    history = dedupe_history_turns(history, collapse_duplicate_user=True)
    if not history:
        return []
    assistant_kept = 0
    kept: List[dict] = []
    for row in reversed(history):
        role = row.get("role")
        if role == "assistant":
            if assistant_kept >= keep_assistant_turns:
                continue
            content = sanitize_turn_content(row.get("content"), role="assistant")
            if len(content) > max_assistant_chars:
                content = content[:max_assistant_chars] + "…"
            if not content:
                continue
            assistant_kept += 1
            kept.append({"role": "assistant", "content": content})
        elif role == "user":
            content = (row.get("content") or "").strip()
            if content:
                kept.append({"role": "user", "content": content})
    kept.reverse()
    return kept


def is_allowed_l1_key(key: str, allowed: tuple[str, ...]) -> bool:
    k = (key or "").strip()
    if not k:
        return False
    return not allowed or k in allowed


def validate_l1_key(key: str, allowed: tuple[str, ...]) -> str:
    k = (key or "").strip()
    if not k:
        raise ValueError("记忆 key 不能为空")
    if allowed and k not in allowed:
        raise ValueError(
            f"不允许的记忆 key: {k}；允许: {', '.join(allowed)}"
        )
    return k


def is_allowed_l1_value(key: str, value: str) -> bool:
    """过滤 finalize / 抽取误写入的 L1 值。"""
    k = (key or "").strip()
    v = (value or "").strip()
    if not v:
        return False
    if k == "输出格式":
        normalized = v.lower().replace(" ", "")
        if normalized in {"json", "json格式", "结构化", "structured"}:
            return False
    return True


def trim_history_tail_chars(
    history: List[dict], max_chars: int
) -> List[dict]:
    """从尾部保留消息，直到字符预算用尽。"""
    if max_chars <= 0 or not history:
        return []
    kept: List[dict] = []
    used = 0
    for row in reversed(history):
        content = (row.get("content") or "").strip()
        if not content:
            continue
        need = len(content) + 8
        if kept and used + need > max_chars:
            break
        kept.append(row)
        used += need
    kept.reverse()
    return kept


def split_history_for_summary(
    history: List[dict],
    *,
    recent_turns: int,
) -> Tuple[List[dict], List[dict]]:
    """拆成 (较早轮次, 最近 N 轮原文)。"""
    if recent_turns <= 0:
        return history, []
    max_recent_msgs = recent_turns * 2
    if len(history) <= max_recent_msgs:
        return [], history
    split_at = len(history) - max_recent_msgs
    return history[:split_at], history[split_at:]


def _format_turns_for_summary(
    turns: List[dict],
    *,
    user_only: bool = False,
) -> str:
    lines: List[str] = []
    for row in turns:
        role = row.get("role", "?")
        if user_only and role != "user":
            continue
        content = sanitize_turn_content(row.get("content"), role=role)
        if content:
            lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines)


async def build_rolling_summary(
    ctx: RunContext,
    older_turns: List[dict],
    cfg: ChatAgentConfig,
) -> str:
    if not older_turns or not cfg.enable_rolling_summary:
        return ""
    raw = _format_turns_for_summary(
        older_turns, user_only=cfg.rolling_summary_user_only
    )
    if not raw.strip():
        return ""
    max_out = cfg.rolling_summary_max_chars
    if len(raw) <= max_out:
        return f"【更早对话摘要】\n{raw}"

    if cfg.use_llm_rolling_summary and ctx.models is not None:
        try:
            model = ctx.get_model(cfg.summarizer_role)
            user_only_note = (
                "摘要仅基于用户发言，不要引用或编造助手曾说的品牌/型号。"
                if cfg.rolling_summary_user_only
                else ""
            )
            from app.agents.memory.memory_runtime_debug import perf_await

            resp = await perf_await(
                ctx,
                "L1.rolling_summary_llm",
                model.ainvoke(
                    [
                        {
                            "role": "system",
                            "content": (
                                f"将以下对话压缩为不超过 {max_out} 字的摘要，"
                                "保留关键事实与用户意图，使用中文条目。"
                                "不要输出任何 thinking 或 XML 标签。"
                                f"{user_only_note}"
                            ),
                        },
                        {"role": "user", "content": raw[:8000]},
                    ]
                ),
            )
            text = strip_model_reasoning(_extract_text(resp).strip())
            if text:
                return f"【更早对话摘要】\n{text[:max_out]}"
        except Exception:
            pass

    return f"【更早对话摘要】\n{raw[:max_out]}…"


def _format_meta_recall_line(msg: dict, *, scope: str) -> str:
    ts = str(msg.get("ts") or "")
    content = sanitize_turn_content(msg.get("content"), role="user")
    if not content:
        return ""
    preview = content if len(content) <= 200 else content[:200] + "..."
    if scope == "user":
        sid = str(msg.get("session_id") or "")
        prefix = f"[session={sid}] " if sid else ""
        return f"{prefix}[{ts}] user: {preview}"
    return f"[{ts}] user: {preview}"


def _collect_recent_user_messages(
    rows: list[dict],
    *,
    limit: int,
    dedupe: bool = True,
    order: str = "desc",
) -> list[dict]:
    """从消息行中取最近 limit 条不重复 user 发言（去重保留最早一条，默认按时间倒序）。"""
    user_rows = [r for r in rows if str(r.get("role") or "") == "user"]
    user_rows.sort(key=lambda r: str(r.get("ts") or ""))
    unique: list[dict] = []
    seen: set[str] = set()
    for row in user_rows:
        content = sanitize_turn_content(row.get("content"), role="user")
        if not content:
            continue
        key = content.strip()
        if dedupe and key in seen:
            continue
        if dedupe:
            seen.add(key)
        unique.append(row)
    picked = unique[-limit:] if len(unique) > limit else unique
    if order == "desc":
        picked = sorted(picked, key=lambda r: str(r.get("ts") or ""), reverse=True)
    return picked


def _sort_messages_by_ts(messages: list[dict], *, order: str = "desc") -> list[dict]:
    reverse = order == "desc"
    return sorted(messages, key=lambda r: str(r.get("ts") or ""), reverse=reverse)


def _format_meta_recall_body(
    messages: list[dict],
    *,
    scope: str,
    max_chars: int,
) -> str:
    lines = [
        line
        for msg in messages
        if (line := _format_meta_recall_line(msg, scope=scope))
    ]
    if not lines:
        return ""
    body = "\n".join(lines)
    if len(body) <= max_chars:
        return body
    truncated = body[:max_chars]
    last_break = truncated.rfind("\n")
    if last_break > max_chars * 0.6:
        truncated = truncated[:last_break]
    return truncated + "\n[... truncated ...]"


async def _prefetch_meta_recall(
    ctx: RunContext,
    query: str,
    cfg: ChatAgentConfig,
    *,
    strategy: str,
    scope: str,
) -> str:
    """meta/b browse 回忆：按时间倒序列最近 user 发言。"""
    from app.agents.memory.memory_runtime_debug import trace_layer_trigger

    memory = ctx.memory
    if memory is None:
        return ""
    limit = max(
        int(getattr(cfg, "meta_recall_recent_limit", 10) or 10),
        cfg.session_search_prefetch_limit,
    )
    max_chars = cfg.session_search_prefetch_max_chars
    label = "跨会话回忆" if strategy == "browse" or scope == "user" else "会话回忆"
    messages: list[dict] = []

    try:
        if strategy == "browse":
            raw = await memory.session_search(
                query or "browse",
                ctx.request,
                limit=min(limit, 10),
                scope="session",  # type: ignore[arg-type]
                mode="browse",
                sort="newest",
            )
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            if payload.get("mode") == "browse":
                for item in payload.get("sessions") or []:
                    if not isinstance(item, dict):
                        continue
                    sid = str((item.get("session") or {}).get("session_id") or "")
                    for msg in item.get("messages") or []:
                        if not isinstance(msg, dict):
                            continue
                        if str(msg.get("role") or "") != "user":
                            continue
                        messages.append({**msg, "session_id": sid})
            messages = _sort_messages_by_ts(messages, order="desc")[:limit]
        else:
            list_recent = getattr(memory, "list_recent_turns", None)
            if list_recent is None:
                trace_layer_trigger(
                    ctx, "L2", "meta_recall_recent", False, "no_list_recent_turns"
                )
                return ""
            rows = await list_recent(ctx.request, limit=limit * 4)
            messages = _collect_recent_user_messages(
                rows, limit=limit, dedupe=True, order="desc"
            )
    except Exception:
        trace_layer_trigger(ctx, "L2", "meta_recall_recent", False, "error")
        return ""

    display_scope = "user" if strategy == "browse" or scope == "user" else "session"
    body = _format_meta_recall_body(
        messages, scope=display_scope, max_chars=max_chars
    )
    trace_layer_trigger(
        ctx,
        "L2",
        "meta_recall_recent",
        bool(body),
        f"strategy={strategy},scope={scope}",
        data={
            "query_preview": (query or "")[:80],
            "user_message_count": len(messages),
            "chars": len(body),
            "recall_strategy": strategy,
        },
    )
    if not body:
        return ""
    return f"【{label}·最近提问】\n{body}"


async def _prefetch_semantic_recall(
    ctx: RunContext,
    query: str,
    cfg: ChatAgentConfig,
    *,
    scope: str,
) -> str:
    from app.agents.memory.memory_runtime_debug import trace_layer_trigger

    memory = ctx.memory
    if memory is None:
        return ""
    trace_layer_trigger(
        ctx,
        "L2",
        "session_search_prefetch",
        True,
        f"scope={scope}",
        data={"query_preview": (query or "")[:80], "recall_strategy": "semantic"},
    )
    try:
        if is_recall_query(query):
            use_llm_summary = cfg.session_search_llm_on_recall
        else:
            use_llm_summary = cfg.session_search_use_llm_summary
        prefer_user = not is_recall_query(query)
        text = await memory.session_search(
            query,
            ctx.request,
            limit=cfg.session_search_prefetch_limit,
            scope=scope,  # type: ignore[arg-type]
            use_llm_summary=use_llm_summary,
            prefer_user_role=prefer_user,
        )
        from app.agents.prompts.text_sanitize import has_model_reasoning, strip_model_reasoning

        text = strip_model_reasoning(text)
        if has_model_reasoning(text):
            text = ""
        if text and text.strip() and not is_empty_session_search_text(text):
            if is_recall_query(query):
                label = "跨会话回忆" if scope == "user" else "会话回忆"
            else:
                label = "跨会话相关" if scope == "user" else "会话相关检索"
            return (
                f"【{label}】\n"
                f"{text.strip()[: cfg.session_search_prefetch_max_chars]}"
            )
    except Exception:
        pass
    return ""


async def prefetch_session_recall(
    ctx: RunContext,
    query: str,
    cfg: ChatAgentConfig,
    *,
    enabled: bool = True,
    plan: Optional["RetrievalPlan"] = None,
) -> str:
    from app.agents.memory.memory_runtime_debug import trace_layer_trigger
    from app.agents.roles.retrieval_router import build_retrieval_plan

    if not enabled:
        trace_layer_trigger(ctx, "L2", "session_search_prefetch", False, "plan_disabled")
        return ""
    if not cfg.session_search_prefetch:
        trace_layer_trigger(ctx, "L2", "session_search_prefetch", False, "config_off")
        return ""
    if hasattr(cfg, "_plan_skip_session_search") and cfg._plan_skip_session_search:
        trace_layer_trigger(ctx, "L2", "session_search_prefetch", False, "plan_skip")
        return ""
    if ctx.memory is None:
        trace_layer_trigger(ctx, "L2", "session_search_prefetch", False, "no_memory_port")
        return ""

    active_plan = plan or build_retrieval_plan(
        query, cfg, enable_rag=cfg.enable_rag
    )
    if not active_plan.run_session_search:
        trace_layer_trigger(ctx, "L2", "session_search_prefetch", False, "plan_off")
        return ""
    if not cfg.retrieval_orchestration and not is_recall_query(query):
        trace_layer_trigger(ctx, "L2", "session_search_prefetch", False, "not_recall_query")
        return ""

    strategy = active_plan.recall_strategy or "semantic"
    scope = active_plan.recall_scope or resolve_recall_scope(query, cfg)

    if strategy == "meta_recent":
        return await _prefetch_meta_recall(
            ctx, query, cfg, strategy="meta_recent", scope=scope
        )
    if strategy == "browse":
        return await _prefetch_meta_recall(
            ctx, query, cfg, strategy="browse", scope=scope
        )
    if strategy == "semantic":
        return await _prefetch_semantic_recall(ctx, query, cfg, scope=scope)
    trace_layer_trigger(
        ctx, "L2", "session_search_prefetch", False, f"strategy={strategy}"
    )
    return ""


async def prefetch_skill_hints(
    ctx: RunContext,
    query: str,
    cfg: ChatAgentConfig,
    *,
    enabled: bool = True,
) -> str:
    from app.agents.memory.memory_runtime_debug import trace_layer_trigger

    if not enabled:
        trace_layer_trigger(ctx, "L3", "skill_prefetch", False, "plan_disabled")
        return ""
    if not cfg.skill_prefetch or not cfg.enable_skill_tools:
        trace_layer_trigger(ctx, "L3", "skill_prefetch", False, "config_off")
        return ""
    if not cfg.retrieval_orchestration and not is_skill_query(query):
        trace_layer_trigger(ctx, "L3", "skill_prefetch", False, "not_skill_query")
        return ""
    memory = ctx.memory
    if memory is None:
        trace_layer_trigger(ctx, "L3", "skill_prefetch", False, "no_memory_port")
        return ""
    trace_layer_trigger(ctx, "L3", "skill_prefetch", True, "skill_search")
    try:
        hits = await memory.skill_search(query, ctx.request, limit=3)
        if not hits:
            return ""
        lines = [
            f"- {h.skill_id}: {h.title} — {h.summary[:120]}"
            for h in hits
        ]
        return "【可用技能】\n" + "\n".join(lines)
    except Exception:
        return ""


async def prefetch_l4_profile(
    ctx: RunContext,
    query: str,
    cfg: ChatAgentConfig,
    *,
    enabled: bool = True,
) -> str:
    from app.agents.memory.memory_runtime_debug import trace_layer_trigger

    if not enabled:
        trace_layer_trigger(ctx, "L4", "profile_prefetch", False, "plan_disabled")
        return ""
    if not cfg.l4_profile_prefetch or not cfg.enable_l4_tools:
        trace_layer_trigger(ctx, "L4", "profile_prefetch", False, "config_off")
        return ""
    if not cfg.retrieval_orchestration and not is_l4_query(query):
        trace_layer_trigger(ctx, "L4", "profile_prefetch", False, "not_l4_query")
        return ""
    memory = ctx.memory
    if memory is None or not hasattr(memory, "fetch_profile_facts"):
        trace_layer_trigger(ctx, "L4", "profile_prefetch", False, "no_memory_port")
        return ""
    trace_layer_trigger(ctx, "L4", "profile_prefetch", True, "fetch_profile_facts")
    try:
        facts = await memory.fetch_profile_facts(
            ctx.request.tenant_id, ctx.request.user_id
        )
        if not facts:
            return ""
        lines: List[str] = []
        for f in facts[:8]:
            if isinstance(f, dict):
                key = f.get("key") or ""
                val = f.get("value") or ""
            else:
                key = getattr(f, "key", "") or ""
                val = getattr(f, "value", "") or ""
            if key:
                lines.append(f"- {key}: {val}")
        if not lines:
            return ""
        return "【外部画像 L4】\n" + "\n".join(lines)
    except Exception:
        return ""


async def prepare_session_context(
    ctx: RunContext,
    history: List[dict],
    user_message: str,
    cfg: ChatAgentConfig,
    retrieval_plan: Optional["RetrievalPlan"] = None,
) -> SessionContextBuildResult:
    """
    返回 SessionContextBuildResult（含 recall_prefetch 供 RAG 短路判断）。
    """
    from app.agents.roles.retrieval_router import build_retrieval_plan

    plan = retrieval_plan or build_retrieval_plan(
        user_message, cfg, enable_rag=cfg.enable_rag
    )

    from app.agents.memory.memory_runtime_debug import trace_layer_trigger

    history = dedupe_history_turns(history)
    collapse_user = plan.intent == "knowledge"
    older, recent = split_history_for_summary(
        history, recent_turns=cfg.recent_turns
    )
    recent = dedupe_history_turns(
        recent, collapse_duplicate_user=collapse_user
    )
    import time as _time

    _prefetch_t0 = _time.perf_counter()

    async def _run_summary() -> str:
        return await build_rolling_summary(ctx, older, cfg)

    async def _run_recall() -> str:
        return await prefetch_session_recall(
            ctx,
            user_message,
            cfg,
            enabled=plan.run_session_search,
            plan=plan,
        )

    async def _run_skills() -> str:
        return await prefetch_skill_hints(
            ctx, user_message, cfg, enabled=plan.run_skill
        )

    async def _run_l4() -> str:
        return await prefetch_l4_profile(
            ctx, user_message, cfg, enabled=plan.run_l4
        )

    summary, recall, skills, l4 = await asyncio.gather(
        _run_summary(),
        _run_recall(),
        _run_skills(),
        _run_l4(),
    )
    from app.agents.memory.memory_runtime_debug import perf_mark

    _prefetch_ms = (_time.perf_counter() - _prefetch_t0) * 1000
    perf_mark(
        ctx,
        "CTX.prefetch_parallel",
        _prefetch_ms,
        intent=plan.intent,
    )
    if plan.run_session_search:
        from app.agents.memory.memory_metrics import record_session_search

        recall_text = (recall or "").strip()
        if not recall_text:
            prefetch_hit = False
        else:
            lower = recall_text.lower()
            prefetch_hit = not any(m in lower for m in _EMPTY_RECALL_MARKERS)
        record_session_search(ctx, hit=prefetch_hit)
    trace_layer_trigger(
        ctx,
        "L1",
        "rolling_summary",
        bool(summary),
        "llm_or_trunc" if summary else "no_older_turns",
        data={"older_turns": len(older), "summary_chars": len(summary or "")},
    )
    trace_layer_trigger(
        ctx,
        "L2",
        "session_search_prefetch",
        bool(recall) and plan.run_session_search,
        "parallel_prefetch",
        data={"enabled": plan.run_session_search},
    )
    if plan.run_skill:
        trace_layer_trigger(
            ctx, "L3", "skill_prefetch", bool(skills), "parallel_prefetch"
        )
    if plan.run_l4:
        trace_layer_trigger(
            ctx, "L4", "profile_prefetch", bool(l4), "parallel_prefetch"
        )

    trimmed = trim_history_tail_chars(recent, cfg.max_history_chars)
    if plan.intent == "knowledge":
        trimmed = compress_history_for_knowledge(trimmed)
    # ── P2: 意图动态预算：知识/寒暄减少历史、回忆保持完整 ──
    _intent = plan.intent
    if _intent in ("knowledge", "chitchat", "recall_and_knowledge"):
        _budget_chars = int(cfg.max_history_chars * 0.6)
        trimmed = trim_history_tail_chars(trimmed, _budget_chars)
    elif _intent == "recall":
        pass  # 回忆意图保持完整历史
    extras: List[str] = []
    if summary:
        extras.append(summary)
    if recall:
        extras.append(recall)
    if skills:
        extras.append(skills)
    if l4:
        extras.append(l4)
    extra_text = "\n\n".join(extras)
    agent_debug(
        "CTX-BUILD",
        "context_builder.prepare_session_context",
        "Context 组装完成",
        {
            "history_in": len(history),
            "older_turns": len(older),
            "recent_in": len(recent),
            "trimmed_out": len(trimmed),
            "has_summary": bool(summary),
            "has_recall_prefetch": bool(recall),
            "has_skill_prefetch": bool(skills),
            "has_l4_prefetch": bool(l4),
            "extra_chars": len(extra_text),
            "recall_query": is_recall_query(user_message),
            "recall_scope": resolve_recall_scope(user_message, cfg),
            "rag_gating_would_run": should_run_rag(user_message, cfg),
            "retrieval_plan": plan.to_debug_dict(),
            "summary_preview": (summary or "")[:200],
            "recall_preview": (recall or "")[:300],
            "skills_preview": (skills or "")[:300],
            "l4_preview": (l4 or "")[:300],
        },
    )
    return SessionContextBuildResult(
        trimmed_history=trimmed,
        extra_text=extra_text,
        recall_prefetch=recall,
    )


def _extract_text(response: Any) -> str:
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        return (getattr(msg, "content", None) or "").strip()
    if isinstance(response, str):
        return response.strip()
    return ""


async def extract_l1_facts_from_session(
    ctx: RunContext,
    turns: List[dict],
    cfg: ChatAgentConfig,
) -> List[dict[str, str]]:
    """会话结束：从 L2 抽取结构化 KV，供 pending L1。返回项含 confidence。"""
    if not cfg.enable_l1_extract_on_finalize or not turns:
        return []
    if not cfg.use_llm_l1_extract or ctx.models is None:
        return []

    transcript = _format_turns_for_summary(turns[-20:])
    if len(transcript) < 20:
        return []

    prompt_keys = ", ".join(cfg.l1_extract_allowed_keys) or "称呼,语言,输出格式"
    try:
        model = ctx.get_model(cfg.summarizer_role)
        resp = await model.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "从对话中提取用户长期偏好，仅输出 JSON 数组 "
                        '[{"key":"...","value":"...","confidence":0.0-1.0}]。'
                        f"只允许 key: {prompt_keys}。"
                        "无明确信息则输出 []。"
                        "输出格式 仅当用户明确要求 JSON/结构化回复时才提取，"
                        "勿因 assistant 使用 JSON 而提取。"
                        "confidence: 明确事实(姓名/称呼/语言)≥0.9，合理推断≥0.6，不确定<0.6。"
                    ),
                },
                {"role": "user", "content": transcript[:6000]},
            ]
        )
        text = _extract_text(resp)
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return []
        items = json.loads(text[start : end + 1])
        allowed = set(cfg.l1_extract_allowed_keys)
        out: List[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            confidence = float(item.get("confidence", 0.7))
            if (
                key
                and value
                and (not allowed or key in allowed)
                and is_allowed_l1_value(key, value)
            ):
                out.append({"key": key, "value": value, "confidence": str(confidence)})
        return out[: cfg.l1_extract_max_items]
    except Exception:
        return []
