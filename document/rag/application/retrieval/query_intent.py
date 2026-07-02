# -*- coding: utf-8 -*-
"""检索前：维护/保养类 query 意图与 metadata 标签解析。"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

# query 侧：命中则视为「机身/耗材维护保养」意图
MAINTENANCE_INTENT_KEYWORDS: Tuple[str, ...] = (
    "保养",
    "维护",
    "机身清洁",
    "机身保养",
    "清洁保养",
    "清洁机器人",
    "擦拭机身",
    "保养机身",
    "维护机身",
    "尘盒",
    "主刷",
    "边刷",
    "滤网",
    "断开电源",
    "耗材",
)

# 建库 metadata 规则 maintenance 打标；历史索引 fallback
MAINTENANCE_RETRIEVAL_TAGS: Tuple[str, ...] = (
    "maintenance",
    "file:维护保养",
)

_MAINTENANCE_SOURCE_MARKERS: Tuple[str, ...] = (
    "维护保养",
    "维护保养.txt",
)

_FAQ_SOURCE_MARKERS: Tuple[str, ...] = (
    "100问",
    "扫地机器人100问",
    "选购指南",
    "常见问题",
    "faq",
)

_NUMBERED_FAQ_RE = re.compile(r"^\s*\d+\.")

_TEMPLATE_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"[，,。！？?\s]*应注意什么[事项]?[？?]?",
        r"[，,。！？?\s]*需要注意什么[事项]?[？?]?",
        r"[，,。！？?\s]*该注意什么[事项]?[？?]?",
        r"[，,。！？?\s]*要注意什么[事项]?[？?]?",
        r"[，,。！？?\s]*有哪些注意事项[？?]?",
        r"[，,。！？?\s]*有什么注意事项[？?]?",
    )
)

_MAINTENANCE_ENTITY_TERMS: Tuple[str, ...] = (
    "维护保养",
    "机身",
    "断开电源",
    "尘盒",
    "主刷",
    "边刷",
    "滤网",
    "传感器",
    "充电触点",
)


def strip_query_templates(query: str) -> str:
    """去掉「应注意什么」等泛化问句模板，保留主题词。"""
    text = (query or "").strip()
    if not text:
        return text
    for pat in _TEMPLATE_PATTERNS:
        text = pat.sub("", text)
    text = re.sub(r"[？?]+$", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip("，,。；; \t")


def has_faq_template_in_query(query: str) -> bool:
    """query 是否含 FAQ 泛化问句模板（剥离前检测）。"""
    text = (query or "").strip()
    if not text:
        return False
    for pat in _TEMPLATE_PATTERNS:
        if pat.search(text):
            return True
    return False


_FAQ_EVIDENCE_TEMPLATE_RE = re.compile(
    r"需要注意什么|应注意什么|要注意什么|该注意什么|有哪些注意事项|有什么注意事项"
)


def is_faq_template_evidence(content: str) -> bool:
    return bool(_FAQ_EVIDENCE_TEMPLATE_RE.search(content or ""))


def _source_path_text(metadata: Optional[dict]) -> str:
    return str(
        (metadata or {}).get("source_path")
        or (metadata or {}).get("ocr_source_path")
        or ""
    )


def is_faq_source(metadata: Optional[dict], content: str = "") -> bool:
    """非维护保养手册来源的 FAQ/选购类条目（100问、选购指南、编号问答等）。"""
    if is_maintenance_source(metadata):
        return False
    source = _source_path_text(metadata).lower()
    if any(m.lower() in source for m in _FAQ_SOURCE_MARKERS):
        return True
    tags_raw = (metadata or {}).get("tags") or (metadata or {}).get("tags_csv") or ""
    tags_text = (
        ",".join(str(t) for t in tags_raw)
        if isinstance(tags_raw, (list, tuple, set))
        else str(tags_raw)
    ).lower()
    if any(f"file:{m.lower()}" in tags_text for m in _FAQ_SOURCE_MARKERS if m != "faq"):
        return True
    text = (content or "").strip()
    if text and _NUMBERED_FAQ_RE.match(text):
        return True
    return False


def evidence_routing_score(evidence) -> float:
    """用于排序的有效分：优先 routing_score，其次 rerank_score。"""
    meta = getattr(evidence, "metadata", None) or {}
    raw = meta.get("routing_score")
    if raw is not None:
        return float(raw)
    raw = meta.get("rerank_score")
    if raw is not None:
        return float(raw)
    return float(getattr(evidence, "score", None) or 0.0)


def apply_post_rerank_maintenance_routing(
    evidences: Sequence,
    query: str,
    *,
    maintenance_boost: float = 0.18,
    faq_penalty: float = 0.12,
) -> list:
    """维护意图：rerank 后对 maintenance 加分、非 maintenance FAQ 降权并重排。"""
    if not evidences or not detect_maintenance_intent(query):
        return list(evidences)
    from dataclasses import replace

    from core.domain.evidence import Evidence

    adjusted: list = []
    for ev in evidences:
        if not isinstance(ev, Evidence):
            adjusted.append(ev)
            continue
        meta = dict(getattr(ev, "metadata", None) or {})
        base = evidence_routing_score(ev)
        delta = 0.0
        if is_maintenance_source(meta):
            delta += maintenance_boost
        elif is_faq_source(meta, ev.content or "") or (
            is_faq_template_evidence(ev.content or "") and not is_maintenance_source(meta)
        ):
            delta -= faq_penalty
        routing = max(0.0, min(1.0, base + delta))
        meta["routing_score"] = routing
        meta["routing_delta"] = delta
        adjusted.append(
            replace(ev, score=routing, metadata=meta)
        )
    adjusted.sort(key=evidence_routing_score, reverse=True)
    return adjusted


def demote_faq_template_evidences(evidences: Sequence, query: str) -> list:
    """维护意图：将 FAQ 模板 chunk 降到队尾（routing 后兜底，不删除）。"""
    if not evidences or not detect_maintenance_intent(query):
        return list(evidences)
    good = []
    faq = []
    for ev in evidences:
        content = getattr(ev, "content", None) or ""
        meta = getattr(ev, "metadata", None) or {}
        if (
            is_faq_source(meta, content)
            or is_faq_template_evidence(content)
        ) and not is_maintenance_source(meta):
            faq.append(ev)
        else:
            good.append(ev)
    return good + faq


def detect_maintenance_intent(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    stripped = strip_query_templates(q)
    for kw in MAINTENANCE_INTENT_KEYWORDS:
        if kw in q or kw in stripped:
            return True
    if "清洁" in q and "机器人" in q and any(
        t in q for t in ("机身", "保养", "维护", "尘盒", "主刷", "耗材")
    ):
        return True
    return False


def resolve_retrieval_tags(query: str) -> Tuple[List[str], str]:
    """维护意图时返回优先 metadata 标签（any 匹配）。"""
    if detect_maintenance_intent(query):
        return list(MAINTENANCE_RETRIEVAL_TAGS), "any"
    return [], "any"


def is_maintenance_source(metadata: Optional[dict]) -> bool:
    """维护保养手册来源：仅认文件路径或 file:维护保养，不认内容关键词打的 maintenance 标签。"""
    if not metadata:
        return False
    source = _source_path_text(metadata)
    if any(m in source for m in _MAINTENANCE_SOURCE_MARKERS):
        return True
    tags_raw = metadata.get("tags") or metadata.get("tags_csv") or ""
    tags_text = (
        ",".join(str(t) for t in tags_raw)
        if isinstance(tags_raw, (list, tuple, set))
        else str(tags_raw)
    )
    return "file:维护保养" in tags_text


def boost_maintenance_evidences(
    evidences: Sequence,
    query: str,
    *,
    boost: float = 0.12,
) -> list:
    """维护意图时对维护保养来源 chunk 加权（软路由）。"""
    if not evidences or not detect_maintenance_intent(query):
        return list(evidences)
    from dataclasses import replace

    from core.domain.evidence import Evidence

    out = []
    for ev in evidences:
        meta = getattr(ev, "metadata", None) or {}
        if is_maintenance_source(meta) and isinstance(ev, Evidence):
            base = float(ev.score or 0.0)
            ev = replace(ev, score=base + boost)
        out.append(ev)
    out.sort(key=lambda e: float(getattr(e, "score", None) or 0.0), reverse=True)
    return out


def maintenance_entity_suffix(query: str) -> str:
    """为改写 query 追加维护实体词（尚未包含的）。"""
    stripped = strip_query_templates(query)
    missing = [t for t in _MAINTENANCE_ENTITY_TERMS if t not in stripped]
    if not missing:
        return ""
    return " ".join(missing[:6])
