# -*- coding: utf-8 -*-
"""LangGraph + Hermes 记忆图状态：工作记忆字段与 prepare 补丁。"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

from core.composition.run_context import RunContext
from core.ports.memory import MemoryDelta

from app.agents.orchestration.chat_config import ChatAgentConfig

MemoryPath = Literal["path_a", "path_b"]

_PENDING_DELTA_KEY = "_pending_memory_delta"


class PendingMemoryDeltaDict(TypedDict, total=False):
    key: str
    value: str
    source: str
    require_hitl: bool


class ChatGraphStateRequired(TypedDict):
    """每轮必填的图状态（Hermes 热记忆 + RAG 摘要）。"""

    user_input: str
    messages: Annotated[list, add_messages]
    assistant_text: str
    evidence_count: int
    rag_empty: bool
    memory_snapshot_hash: str
    evidences_summary: list
    memory_summary: dict


class ChatGraphState(ChatGraphStateRequired, total=False):
    """LangGraph 工作记忆扩展（Router / L1 pending / L0 标记）。"""

    retrieval_intent: str
    pending_remember: Optional[str]
    pending_memory_delta: list[PendingMemoryDeltaDict]
    l0_applied: bool
    memory_path: MemoryPath


def resolve_memory_path(chat_cfg: ChatAgentConfig) -> MemoryPath:
    return "path_b" if chat_cfg.enable_memory_tools else "path_a"


def delta_to_dict(
    delta: MemoryDelta,
    *,
    require_hitl: bool = True,
) -> PendingMemoryDeltaDict:
    return {
        "key": delta.key,
        "value": delta.value,
        "source": delta.source,
        "require_hitl": require_hitl,
    }


def _extra_dict(ctx: RunContext) -> dict[str, Any]:
    extra = getattr(ctx, "extra", None)
    return extra if isinstance(extra, dict) else {}


def pending_memory_delta_from_ctx(ctx: RunContext) -> list[PendingMemoryDeltaDict]:
    raw = _extra_dict(ctx).get(_PENDING_DELTA_KEY)
    if not isinstance(raw, list):
        return []
    out: list[PendingMemoryDeltaDict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        if not key or not value:
            continue
        out.append(
            {
                "key": key,
                "value": value,
                "source": str(item.get("source") or "user"),
                "require_hitl": bool(item.get("require_hitl", True)),
            }
        )
    return out


def append_pending_memory_delta(
    ctx: RunContext,
    delta: MemoryDelta | PendingMemoryDeltaDict,
    *,
    require_hitl: bool = True,
) -> list[PendingMemoryDeltaDict]:
    """会话内累积 L1 待写入 delta（GraphState + ctx.extra）。"""
    extra = getattr(ctx, "extra", None)
    if not isinstance(extra, dict):
        return []
    if isinstance(delta, MemoryDelta):
        entry = delta_to_dict(delta, require_hitl=require_hitl)
    else:
        entry = dict(delta)
        entry.setdefault("require_hitl", require_hitl)
    current = pending_memory_delta_from_ctx(ctx)
    merged = {d["key"]: d for d in current}
    merged[entry["key"]] = entry  # 同 key 以最后一次为准
    updated = list(merged.values())
    extra[_PENDING_DELTA_KEY] = updated
    latest = f"{entry['key']}={entry['value']}"
    extra["pending_remember"] = latest
    store_graph_memory_snapshot(
        ctx,
        {
            "pending_memory_delta": updated,
            "pending_remember": latest,
        },
    )
    return updated


def clear_pending_memory_delta(ctx: RunContext) -> None:
    extra = _extra_dict(ctx)
    if not extra:
        return
    extra.pop(_PENDING_DELTA_KEY, None)
    extra.pop("pending_remember", None)
    store_graph_memory_snapshot(
        ctx,
        {"pending_memory_delta": [], "pending_remember": None},
    )


def working_memory_from_ctx(ctx: RunContext) -> dict[str, Any]:
    """从 RunContext.extra 提取 Router / pending L1 等工作记忆。"""
    extra = _extra_dict(ctx)
    if not extra:
        return {}
    out: dict[str, Any] = {}
    intent = extra.get("retrieval_intent")
    if intent is not None:
        out["retrieval_intent"] = str(intent)
    pending = extra.get("pending_remember")
    if pending:
        out["pending_remember"] = str(pending)
    deltas = pending_memory_delta_from_ctx(ctx)
    if deltas:
        out["pending_memory_delta"] = deltas
    return out


def build_prepare_state_patch(
    *,
    user_input: str,
    lc_messages: list,
    ev_count: int,
    rag_empty: bool,
    mem_hash: str,
    evidences_summary: list,
    memory_summary: dict,
    ctx: RunContext,
    chat_cfg: ChatAgentConfig,
) -> dict[str, Any]:
    """prepare 节点返回的状态补丁（含 REMOVE_ALL_MESSAGES 由调用方注入）。"""
    from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage

    wm = working_memory_from_ctx(ctx)
    return {
        "user_input": user_input,
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *lc_messages],
        "evidence_count": ev_count,
        "rag_empty": rag_empty,
        "memory_snapshot_hash": mem_hash,
        "evidences_summary": evidences_summary or [],
        "memory_summary": memory_summary or {},
        "retrieval_intent": wm.get("retrieval_intent", ""),
        "pending_remember": wm.get("pending_remember"),
        "pending_memory_delta": wm.get("pending_memory_delta") or [],
        "memory_path": resolve_memory_path(chat_cfg),
        "l0_applied": False,
    }


def l0_compress_triggered(ctx: RunContext, compress_count_before: int) -> bool:
    extra = _extra_dict(ctx)
    after = int(extra.get("_l0_compress_count") or 0)
    return after > compress_count_before


_GRAPH_SNAPSHOT_KEY = "_graph_memory_snapshot"


def memory_working_fields_from_state(state: dict[str, Any] | None) -> dict[str, Any]:
    """从 LangGraph 终态提取 Hermes 工作记忆字段。"""
    if not state:
        return {}
    deltas = state.get("pending_memory_delta")
    return {
        "retrieval_intent": str(state.get("retrieval_intent") or ""),
        "memory_path": str(state.get("memory_path") or ""),
        "pending_remember": state.get("pending_remember"),
        "pending_memory_delta": list(deltas) if isinstance(deltas, list) else [],
        "l0_applied": bool(state.get("l0_applied")),
    }


def merge_memory_summary_with_state(
    memory_summary: dict[str, Any] | None,
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    """将 GraphState 工作记忆字段并入 memory_summary（供 Web / SSE）。"""
    merged = dict(memory_summary or {})
    merged.update(memory_working_fields_from_state(state))
    return merged


def store_graph_memory_snapshot(ctx: RunContext, fields: dict[str, Any]) -> None:
    """写入 RunContext.extra，供观测面板与流式 meta 读取。"""
    extra = _extra_dict(ctx)
    if not extra:
        return
    snap = dict(extra.get(_GRAPH_SNAPSHOT_KEY) or {})
    snap.update(fields)
    extra[_GRAPH_SNAPSHOT_KEY] = snap


def read_graph_memory_snapshot(ctx: RunContext) -> dict[str, Any]:
    extra = _extra_dict(ctx)
    raw = extra.get(_GRAPH_SNAPSHOT_KEY)
    return dict(raw) if isinstance(raw, dict) else {}


def update_graph_memory_snapshot(ctx: RunContext, **fields: Any) -> None:
    store_graph_memory_snapshot(ctx, fields)


def rehydrate_working_memory_to_ctx(
    ctx: RunContext,
    state: dict[str, Any] | None,
) -> None:
    """Checkpointer 恢复：将 GraphState 工作记忆写回 ctx.extra。"""
    if not state:
        return
    extra = getattr(ctx, "extra", None)
    if not isinstance(extra, dict):
        return
    fields = memory_working_fields_from_state(state)
    if fields.get("retrieval_intent"):
        extra["retrieval_intent"] = fields["retrieval_intent"]
    deltas = fields.get("pending_memory_delta") or []
    if deltas:
        extra[_PENDING_DELTA_KEY] = list(deltas)
    if fields.get("pending_remember"):
        extra["pending_remember"] = fields["pending_remember"]
    store_graph_memory_snapshot(ctx, fields)


def sync_pending_deltas_from_hot(ctx: RunContext) -> int:
    """从 L1 hot pending 队列对齐 ctx.extra（进程重启 / 多入口恢复）。"""
    memory = getattr(ctx, "memory", None)
    if memory is None:
        return 0
    hot = getattr(memory, "_hot", None)
    if hot is None or not hasattr(hot, "list_pending_deltas"):
        return 0
    try:
        queued = hot.list_pending_deltas(
            ctx.request.tenant_id, ctx.request.user_id
        )
    except Exception:
        return 0
    if not queued:
        return 0
    for delta in queued:
        append_pending_memory_delta(ctx, delta, require_hitl=True)
    return len(queued)


def _parse_kv_content(content: str | None) -> tuple[str, str] | None:
    text = (content or "").strip()
    if ": " not in text:
        return None
    key, value = text.split(": ", 1)
    key, value = key.strip(), value.strip()
    if key and value:
        return key, value
    return None


def record_l1_write(
    ctx: RunContext,
    *,
    key: str,
    value: str,
    source: str = "user",
    require_hitl: bool = True,
) -> None:
    """统一记录 L1 写入到 GraphState pending 追踪。"""
    if not key or not value:
        return
    append_pending_memory_delta(
        ctx,
        MemoryDelta(key=key, value=value, source=source),
        require_hitl=require_hitl,
    )


def record_memory_tool_result(
    ctx: RunContext,
    result: dict[str, Any],
    *,
    action: str = "",
    target: str = "user",
    content: str | None = None,
) -> None:
    """Hermes memory 工具结果 → GraphState delta。"""
    if not isinstance(result, dict) or not result.get("success"):
        return
    require_hitl = bool(result.get("staged"))
    parsed = _parse_kv_content(content)
    if parsed and action == "add":
        record_l1_write(
            ctx,
            key=parsed[0],
            value=parsed[1],
            source="memory_tool",
            require_hitl=require_hitl,
        )
        return
    if require_hitl:
        entry: PendingMemoryDeltaDict = {
            "key": f"__{target}_{action}",
            "value": (content or str(result.get("message") or "staged"))[:200],
            "source": "memory_tool",
            "require_hitl": True,
        }
        append_pending_memory_delta(ctx, entry, require_hitl=True)


async def flush_graph_pending_deltas_on_finalize(ctx: RunContext) -> int:
    """
    会话 end：将 GraphState 累积的 pending delta 写入 MemoryPort（若尚未 queue）。
    finalize_session 会 flush hot pending；此处确保图状态与 Port 对齐并清空。
    """
    deltas = pending_memory_delta_from_ctx(ctx)
    if not deltas:
        return 0
    memory = getattr(ctx, "memory", None)
    if memory is None:
        clear_pending_memory_delta(ctx)
        return 0
    hot = getattr(memory, "_hot", None)
    existing_keys: set[str] = set()
    if hot is not None and hasattr(hot, "list_pending_deltas"):
        try:
            queued = hot.list_pending_deltas(
                ctx.request.tenant_id, ctx.request.user_id
            )
            existing_keys = {d.key for d in queued}
        except Exception:
            existing_keys = set()
    written = 0
    for item in deltas:
        key = item["key"]
        if str(key).startswith("__"):
            continue
        if key in existing_keys:
            continue
        await memory.update_prompt_memory(
            ctx.request,
            MemoryDelta(
                key=key,
                value=item["value"],
                source=str(item.get("source") or "user"),
            ),
            require_hitl=bool(item.get("require_hitl", True)),
        )
        written += 1
    clear_pending_memory_delta(ctx)
    return written
