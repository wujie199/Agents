"""prompt_builder 单元测试。"""

from __future__ import annotations

from core.domain.evidence import Evidence, EvidenceBundle, SourceType

from app.agents.prompts.prompt_builder import filter_evidence_bundle, format_evidence_bundle


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        evidences=[
            Evidence(id="1", content="高相关", source_type=SourceType.VECTOR, score=0.9),
            Evidence(id="2", content="低相关", source_type=SourceType.VECTOR, score=0.1),
        ],
        empty=False,
    )


def test_filter_evidence_bundle_by_score():
    filtered = filter_evidence_bundle(_bundle(), min_score=0.5)
    assert len(filtered.evidences) == 1
    assert filtered.evidences[0].id == "1"


def test_filter_evidence_bundle_min_keep():
    filtered = filter_evidence_bundle(_bundle(), min_score=0.5, min_keep=2)
    assert len(filtered.evidences) == 2


def test_filter_evidence_bundle_all_below_threshold():
    filtered = filter_evidence_bundle(_bundle(), min_score=0.95)
    assert filtered.empty
    assert filtered.error_code == "rag_below_min_score"


def test_format_evidence_bundle_respects_min_score():
    text = format_evidence_bundle(_bundle(), min_score=0.5)
    assert "高相关" in text
    assert "低相关" not in text
