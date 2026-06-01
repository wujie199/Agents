"""规则/关键词 metadata 打标。"""

from pathlib import Path

import pytest

from core.ports.ingest import IngestResult, IngestStatus
from document.rag.adapters.metadata.rule_keyword import (
    RuleKeywordMetadataEnricher,
    TaggingRule,
    load_tagging_rules,
)
from document.rag.application.metadata.pipeline import apply_metadata_enrichment
from document.rag.config import MetadataConfig, RagPipelineConfig


@pytest.fixture
def rules_path():
    return str(Path("config") / "metadata_tagging.yml")


def test_load_rules(rules_path):
    rules, ext = load_tagging_rules(rules_path)
    assert len(rules) >= 3
    assert "pdf" in ext


def test_contract_keywords_match(rules_path):
    enricher = RuleKeywordMetadataEnricher(rules_path=rules_path, tag_filename=False)
    ingest = IngestResult(
        content="本合同约定甲方与乙方的违约责任及合同编号。",
        metadata={"doc_id": "d1"},
        status=IngestStatus.SUCCESS,
    )
    out = enricher.enrich(ingest, doc_format="pdf")
    assert "合同" in out.metadata["tags"]
    assert "contract" in out.metadata["matched_rules"]


def test_match_all_mode():
    enricher = RuleKeywordMetadataEnricher(
        rules=[
            TaggingRule(
                name="strict",
                tags=["严格"],
                keywords=["甲方", "乙方"],
                match="all",
            )
        ],
        tag_filename=False,
    )
    partial = enricher.enrich(
        IngestResult(content="仅提到甲方。", metadata={}),
    )
    assert "strict" not in partial.metadata.get("matched_rules", [])

    full = enricher.enrich(
        IngestResult(content="甲方与乙方签字。", metadata={}),
    )
    assert "strict" in full.metadata["matched_rules"]
    assert "严格" in full.metadata["tags"]


def test_apply_metadata_enrichment_disabled():
    from core.ports.ingest import DocumentFormat

    cfg = RagPipelineConfig(metadata=MetadataConfig(enabled=False))
    ingest = IngestResult(content="甲方 乙方", metadata={})
    out = apply_metadata_enrichment(ingest, DocumentFormat.PDF, cfg)
    assert "metadata_tagged" not in out.metadata
