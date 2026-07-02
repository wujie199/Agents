# -*- coding: utf-8 -*-
"""对话 prompt 组装：L1 快照 + RAG 证据（标注 untrusted）。"""

from __future__ import annotations

from core.domain.evidence import DegradedReason, Evidence, EvidenceBundle
from document.rag.application.retrieval.query_intent import evidence_routing_score

from app.agents.prompts.text_sanitize import sanitize_turn_content


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

EVIDENCE_EVAL_GROUNDING_RULES = (
    "【评测回答约束】当上方存在检索片段时：\n"
    "1. 只回答用户当前问题，不要补充未被问及的关联问题或额外排查步骤。\n"
    "2. 优先采用与问题最直接匹配的第一条检索片段（[1]），仅在其信息不足时再参考后续片段。\n"
    "3. 完整保留片段中的具体数字、尺寸、时间、步骤顺序与条件限制，不要省略或改写。\n"
    "4. 用简洁的一段或 1-3 条短句作答，避免多级标题、冗长列举或与问题无关的扩展内容。"
)

RAG_MISS_GROUNDING_RULES = (
    "【知识库检索说明】本轮未在知识库中找到与问题相关的可靠片段。\n"
    "请明确告知用户「当前知识库中未找到相关资料」，不要编造产品操作步骤、型号、政策或通用教程。\n"
    "可简要说明可能原因（未入库、表述不匹配），并建议用户换关键词或上传相关文档。"
)

RECALL_TOOL_HINT = (
    "【工具提示】预检索未找到相关历史。若需回忆过往对话，请调用 session_search 工具"
    "（scope=session 查本会话，scope=user 查跨会话）。"
)

PROFILE_NAME_HINT = (
    "【工具提示】用户正在告知或更正姓名/称呼。请用自然语言确认。"
    "若尚未写入记忆，请调用 remember_user_fact（key=姓名 或 称呼）。"
    "系统可能已自动加入待确认列表，可提示用户输入 /confirm 确认写入 L1。"
)

SKILL_TOOL_HINT = (
    "【工具提示】用户要使用 L3 技能/流程。请先 skill_search 查找匹配技能，"
    "再 run_skill(skill_id, inputs) 执行；inputs 为 JSON 对象字符串。"
    "若上方【可用技能】已有候选，可直接选用其 skill_id。"
)


def _effective_evidence_score(evidence: Evidence) -> float:
    return evidence_routing_score(evidence)


def filter_evidence_bundle(
    bundle: EvidenceBundle,
    *,
    min_score: float = 0.0,
    min_keep: int = 0,
) -> EvidenceBundle:
    """按相关性分数过滤证据；可选保留 top-N 以免高分片段过少。"""
    if not bundle.evidences:
        return bundle
    ranked = sorted(bundle.evidences, key=_effective_evidence_score, reverse=True)
    if min_score <= 0:
        return EvidenceBundle(
            evidences=ranked,
            plan=bundle.plan,
            empty=False,
            degraded_reason=bundle.degraded_reason,
            error_code=bundle.error_code,
        )
    kept = [ev for ev in ranked if _effective_evidence_score(ev) >= min_score]
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


def format_rag_miss_notice(
    bundle: EvidenceBundle | None = None,
    *,
    strict: bool = True,
) -> str:
    """知识类问题检索未命中时注入的反幻觉说明。"""
    if not strict:
        return ""
    parts = [RAG_MISS_GROUNDING_RULES]
    if bundle is not None and bundle.is_degraded() and bundle.degraded_reason:
        code = f" code={bundle.error_code}" if bundle.error_code else ""
        parts.append(
            f"[检索状态: {bundle.degraded_reason.value}{code}]"
        )
    return "\n".join(parts)


def format_evidence_bundle(
    bundle: EvidenceBundle,
    *,
    max_chars: int = 6000,
    max_items: int = 8,
    min_score: float = 0.0,
    min_keep: int = 0,
    strict_grounding: bool = False,
    eval_strict_answer: bool = False,
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
    if eval_strict_answer and count > 0:
        return f"{body}\n\n{EVIDENCE_EVAL_GROUNDING_RULES}"
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
