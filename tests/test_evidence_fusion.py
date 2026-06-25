# -*- coding: utf-8 -*-
"""证据融合模块单元测试。"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.agents.roles.evidence_fusion import (
    FusedEvidence,
    fuse_evidence,
    _content_fingerprint,
    _EvidenceItem,
    _parse_recall_items,
    _parse_rag_items,
)
from app.agents.orchestration.chat_config import ChatAgentConfig


def _cfg(**kw) -> ChatAgentConfig:
    defaults = dict(
        fusion_recall_weight=0.6,
        fusion_rag_weight=0.4,
    )
    defaults.update(kw)
    return MagicMock(
        fusion_recall_weight=defaults["fusion_recall_weight"],
        fusion_rag_weight=defaults["fusion_rag_weight"],
        spec=ChatAgentConfig,
    )


class TestContentFingerprint:
    def test_empty(self):
        assert _content_fingerprint("") == ""
        assert _content_fingerprint("   ") == ""

    def test_deterministic(self):
        assert _content_fingerprint("hello") == _content_fingerprint("hello")

    def test_normalization(self):
        # 前100字取hash，大小写归一
        fp1 = _content_fingerprint("Hello World")
        fp2 = _content_fingerprint("hello world")
        assert fp1 == fp2

    def test_long_text_truncates(self):
        short = "a" * 100
        long = "a" * 200
        assert _content_fingerprint(short) == _content_fingerprint(long)


class TestFuseEvidence:
    def test_both_empty(self):
        cfg = _cfg()
        result = fuse_evidence("", "", cfg)
        assert result.evidence_text == ""
        assert result.recall_count == 0
        assert result.rag_count == 0
        assert result.deduped_count == 0

    def test_recall_only(self):
        cfg = _cfg()
        result = fuse_evidence("我的名字叫张三\n我住在北京朝阳", "", cfg)
        assert result.recall_count == 2
        assert result.rag_count == 0
        assert result.total_count == 2
        assert "【回忆】" in result.evidence_text

    def test_rag_only(self):
        cfg = _cfg()
        result = fuse_evidence("", "知识条目详细描述一\n知识条目详细描述二", cfg)
        assert result.recall_count == 0
        assert result.rag_count == 2
        assert "【知识】" in result.evidence_text

    def test_dedup_same_content(self):
        cfg = _cfg()
        same_line = "用户偏好深色主题模式"
        result = fuse_evidence(same_line, same_line, cfg)
        assert result.deduped_count >= 1

    def test_max_items_limit(self):
        cfg = _cfg()
        recall = "\n".join(f"回忆条目{i}" for i in range(20))
        result = fuse_evidence(recall, "", cfg, max_items=3)
        # evidence_text 行数 <= 3
        lines = [l for l in result.evidence_text.split("\n") if l.strip()]
        assert len(lines) <= 3

    def test_max_chars_limit(self):
        cfg = _cfg()
        recall = "\n".join(f"很长的回忆条目内容编号{i}包含很多字" for i in range(20))
        result = fuse_evidence(recall, "", cfg, max_chars=200)
        # max_chars 截断已考虑 \n 分隔符，输出严格 ≤ max_chars
        assert len(result.evidence_text) <= 200

    def test_weighted_sorting(self):
        """recall_weight > rag_weight 时 recall 排前面。"""
        cfg = _cfg(fusion_recall_weight=0.9, fusion_rag_weight=0.1)
        recall = "重要的回忆内容详情"
        rag = "不太重要的知识内容详情"
        result = fuse_evidence(recall, rag, cfg)
        lines = result.evidence_text.split("\n")
        if len(lines) >= 2:
            assert lines[0].startswith("【回忆】")

    def test_marker_stripping(self):
        cfg = _cfg()
        result = fuse_evidence("【会话回忆】条目内容A\n条目内容B", "", cfg)
        assert result.recall_count == 2

    def test_rag_marker_stripping(self):
        cfg = _cfg()
        result = fuse_evidence("", "【检索证据】证据内容A\n证据内容B", cfg)
        assert result.rag_count == 2


class TestFusedEvidenceTotalCount:
    def test_total(self):
        e = FusedEvidence(evidence_text="x", recall_count=5, rag_count=3, deduped_count=1)
        assert e.total_count == 7
