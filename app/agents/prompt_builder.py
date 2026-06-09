# -*- coding: utf-8 -*-
"""对话 prompt 组装：L1 快照 + RAG 证据（标注 untrusted）。"""

from __future__ import annotations

from core.domain.evidence import DegradedReason, EvidenceBundle

from app.agents.text_sanitize import sanitize_turn_content


EVIDENCE_PREAMBLE = (
    "以下检索片段仅供参考，可能不完整或过时，请勿当作唯一事实依据："
)

EVIDENCE_GROUNDING_RULES = (
    "【回答约束】当上方存在检索片段时：\n"
    "1. 仅依据检索片段与当前用户问题作答，不得编造未出现在片段中的品牌、型号或数据。\n"
    "2. 可基于片段中的对比、限制或负面描述合理归纳（如「需人工操作」「成本较高」可归纳为缺点）；"
    "仅当片段完全无关时再说明「资料中未明确提及」，不要猜测。\n"
    "3. 优先引用片段原文，避免与历史对话中可能错误的 assistant 回复冲突。"
)

RECALL_TOOL_HINT = (
    "【工具提示】预检索未找到相关历史。若需回忆过往对话，请调用 session_search 工具"
    "（scope=session 查本会话，scope=user 查跨会话）。"
)

PROFILE_NAME_HINT = (
    "【工具提示】用户正在告知或更正姓名/称呼。请用自然语言确认，"
    "并调用 remember 工具将 key=姓名 或 称呼 写入长期记忆。"
)


def filter_evidence_bundle(
    bundle: EvidenceBundle,
    *,
    min_score: float = 0.0,
    min_keep: int = 0,
) -> EvidenceBundle:
    """按相关性分数过滤证据；可选保留 top-N 以免高分片段过少。"""
    if not bundle.evidences:
        return bundle
    ranked = sorted(bundle.evidences, key=lambda ev: ev.score, reverse=True)
    if min_score <= 0:
        return bundle
    kept = [ev for ev in ranked if ev.score >= min_score]
    if min_keep > 0 and len(kept) < min_keep:
        seen = {ev.id for ev in kept}
        for ev in ranked:
            if ev.id in seen:
                continue
            kept.append(ev)
            seen.add(ev.id)
            if len(kept) >= min_keep:
                break
    if not kept:
        return EvidenceBundle.empty_bundle(
            bundle.degraded_reason or DegradedReason.PARTIAL_RESULTS,
            "rag_below_min_score",
            plan=bundle.plan,
        )
    return EvidenceBundle(
        evidences=kept,
        plan=bundle.plan,
        empty=False,
        degraded_reason=bundle.degraded_reason,
        error_code=bundle.error_code,
    )


def format_evidence_bundle(
    bundle: EvidenceBundle,
    *,
    max_chars: int = 6000,
    max_items: int = 8,
    min_score: float = 0.0,
    min_keep: int = 0,
    strict_grounding: bool = False,
) -> str:
    """将 EvidenceBundle 格式化为可拼入 system 的文本块。"""
    bundle = filter_evidence_bundle(
        bundle, min_score=min_score, min_keep=min_keep
    )
    if bundle.empty or not bundle.evidences:
        if bundle.is_degraded() and bundle.degraded_reason:
            return (
                f"[检索降级: {bundle.degraded_reason.value}"
                f"{(' code=' + bundle.error_code) if bundle.error_code else ''}]"
            )
        return ""

    lines = [EVIDENCE_PREAMBLE, "---"]
    used = len(lines[0]) + len(lines[1])
    count = 0

    for i, ev in enumerate(bundle.evidences, 1):
        if count >= max_items:
            break
        src = ev.citation or ev.metadata.get("source_path") or ""
        header = f"[{i}] score={ev.score:.4f}"
        if src:
            header += f" source={src}"
        text = (ev.content or "").strip()
        if not text:
            continue
        block = f"{header}\n{text}\n---"
        if used + len(block) > max_chars:
            remain = max_chars - used - len(header) - 10
            if remain > 80:
                block = f"{header}\n{text[:remain]}…\n---"
            else:
                break
        lines.append(block)
        used += len(block)
        count += 1

    body = "\n".join(lines).strip()
    if strict_grounding and count > 0:
        return f"{body}\n\n{EVIDENCE_GROUNDING_RULES}"
    return body


def build_chat_messages(
    *,
    memory_system: str,
    user_message: str,
    evidence_text: str = "",
    session_context_text: str = "",
    history: list[dict] | None = None,
    tool_hints: str = "",
) -> list[dict[str, str]]:
    """组装发往 LLM 的 messages 列表。"""
    system_parts = [memory_system.strip()]
    if session_context_text.strip():
        system_parts.append("")
        system_parts.append(session_context_text.strip())
    if tool_hints.strip():
        system_parts.append("")
        system_parts.append(tool_hints.strip())
    if evidence_text.strip():
        system_parts.append("")
        system_parts.append(evidence_text.strip())

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "\n".join(system_parts)},
    ]

    for row in history or []:
        role = row.get("role")
        content = sanitize_turn_content(row.get("content"), role=role or "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages
