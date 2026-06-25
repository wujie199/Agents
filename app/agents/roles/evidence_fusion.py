# -*- coding: utf-8 -*-
"""证据融合：recall 结果与 RAG 结果互校、去重、权重合并。

当意图为 recall_and_knowledge 时，session_search 和 RAG 并行返回结果，
此模块负责将两路证据融合为统一的 evidence 文本注入 prompt。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

from app.agents.orchestration.chat_config import ChatAgentConfig


@dataclass(frozen=True)
class FusedEvidence:
    """融合后的证据结果。"""

    evidence_text: str
    recall_count: int
    rag_count: int
    deduped_count: int  # 去重移除的条目数

    @property
    def total_count(self) -> int:
        return self.recall_count + self.rag_count - self.deduped_count


def _content_fingerprint(text: str) -> str:
    """对文本做指纹，用于去重（取前 100 字的 hash）。"""
    normalized = (text or "").strip().lower()[:100]
    if not normalized:
        return ""
    return hashlib.md5(normalized.encode("utf-8", errors="replace")).hexdigest()[:12]


@dataclass
class _EvidenceItem:
    source: str  # "recall" | "rag"
    content: str
    score: float  # 0.0-1.0
    citation: str = ""
    fingerprint: str = ""

    @property
    def weighted_score(self) -> float:
        return self.score


def fuse_evidence(
    recall_text: str,
    rag_evidence_text: str,
    cfg: ChatAgentConfig,
    *,
    max_chars: int = 6000,
    max_items: int = 10,
) -> FusedEvidence:
    """将 recall 预检索结果和 RAG 证据文本融合。

    策略：
    1. 解析两路结果为 _EvidenceItem 列表
    2. 指纹去重（recall 和 RAG 引用相同内容时保留高分者）
    3. 加权排序：recall 项 * fusion_recall_weight，rag 项 * fusion_rag_weight
    4. 截断到 max_chars / max_items
    5. 返回融合后的 evidence 文本
    """
    recall_items = _parse_recall_items(recall_text, cfg)
    rag_items = _parse_rag_items(rag_evidence_text, cfg)

    if not recall_items and not rag_items:
        return FusedEvidence(
            evidence_text="",
            recall_count=0,
            rag_count=0,
            deduped_count=0,
        )

    # 去重：同 fingerprint 保留加权分高的
    seen: dict[str, _EvidenceItem] = {}
    deduped = 0
    for item in recall_items + rag_items:
        fp = item.fingerprint
        if not fp:
            seen[f"{item.source}_{id(item)}"] = item
            continue
        if fp in seen:
            existing = seen[fp]
            if item.weighted_score > existing.weighted_score:
                seen[fp] = item
            deduped += 1
        else:
            seen[fp] = item

    # 加权排序
    sorted_items = sorted(
        seen.values(),
        key=lambda x: x.weighted_score,
        reverse=True,
    )

    # 格式化输出
    lines: List[str] = []
    total_chars = 0
    count = 0
    for item in sorted_items:
        if count >= max_items:
            break
        source_label = "【回忆】" if item.source == "recall" else "【知识】"
        citation_part = f"（{item.citation}）" if item.citation else ""
        line = f"{source_label}{citation_part} {item.content}"
        # +1 for the \n separator between lines (not needed for the first line)
        sep = 1 if lines else 0
        if total_chars + sep + len(line) > max_chars:
            break
        lines.append(line)
        total_chars += sep + len(line)
        count += 1

    evidence_text = "\n".join(lines) if lines else ""
    return FusedEvidence(
        evidence_text=evidence_text,
        recall_count=len(recall_items),
        rag_count=len(rag_items),
        deduped_count=deduped,
    )


def _parse_recall_items(text: str, cfg: ChatAgentConfig) -> List[_EvidenceItem]:
    """将 session_search 预检索文本拆分为条目。"""
    if not text or not text.strip():
        return []
    # session_search 结果通常以 "- " 或行分隔
    raw = text.strip()
    # 去掉标记头
    for marker in ("【会话回忆】", "【跨会话回忆】", "【会话相关检索】", "【跨会话相关】"):
        if raw.startswith(marker):
            raw = raw[len(marker):].strip()
            break

    items: List[_EvidenceItem] = []
    for line in raw.split("\n"):
        line = line.strip().lstrip("- ").strip()
        if not line or len(line) < 5:
            continue
        items.append(
            _EvidenceItem(
                source="recall",
                content=line,
                score=cfg.fusion_recall_weight,
                fingerprint=_content_fingerprint(line),
            )
        )
    return items[:20]


def _parse_rag_items(text: str, cfg: ChatAgentConfig) -> List[_EvidenceItem]:
    """将 RAG evidence 文本拆分为条目。"""
    if not text or not text.strip():
        return []
    raw = text.strip()
    # 去掉标记头
    for marker in ("【检索证据】", "【知识库检索】", "【RAG 证据】"):
        if raw.startswith(marker):
            raw = raw[len(marker):].strip()
            break

    items: List[_EvidenceItem] = []
    for line in raw.split("\n"):
        line = line.strip().lstrip("- ").strip()
        if not line or len(line) < 5:
            continue
        items.append(
            _EvidenceItem(
                source="rag",
                content=line,
                score=cfg.fusion_rag_weight,
                fingerprint=_content_fingerprint(line),
            )
        )
    return items[:20]
