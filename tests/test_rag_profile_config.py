"""RAG profile 配置选择与加载测试。"""

from pathlib import Path

import pytest

from document.rag.config import (
    detect_rag_profile_for_path,
    load_rag_pipeline_config,
    resolve_rag_pipeline_config_path,
)
from document.rag.config.rag_yaml import deep_merge_rag_config, load_rag_yaml_document
from document.rag.components.cleaner.factory import build_cleaner_from_rag_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO_ROOT / "config")


def test_deep_merge_nested_dict():
    base = {"ingest": {"mode": "ocr_only", "word_to_pdf": True}, "keep": 1}
    overlay = {"ingest": {"enable_header_footer_dedup": True}, "keep": 2}
    merged = deep_merge_rag_config(base, overlay)
    assert merged["ingest"]["mode"] == "ocr_only"
    assert merged["ingest"]["word_to_pdf"] is True
    assert merged["ingest"]["enable_header_footer_dedup"] is True
    assert merged["keep"] == 2


def test_deep_merge_chunk_pipeline_delta():
    base = {
        "chunk_pipeline": {
            "domain": "general",
            "preserve_faq_pairs": True,
            "child_target_chars": 200,
        }
    }
    overlay = {"chunk_pipeline": {"domain": "faq", "child_target_chars": 400}}
    merged = deep_merge_rag_config(base, overlay)
    assert merged["chunk_pipeline"]["domain"] == "faq"
    assert merged["chunk_pipeline"]["preserve_faq_pairs"] is True
    assert merged["chunk_pipeline"]["child_target_chars"] == 400


def test_faq_profile_merges_base_metadata_rules():
    path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile="faq")
    raw = load_rag_yaml_document(path, config_dir=CONFIG_DIR)
    rules = (raw.get("metadata") or {}).get("rules") or []
    assert any(r.get("name") == "contract" for r in rules)


def test_resolve_profile_paths():
    faq_path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile="faq")
    assert faq_path.endswith("rag.faq.yml")
    contract_path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile="contract")
    assert contract_path.endswith("rag.contract.yml")


def test_detect_rag_profile_for_path():
    assert detect_rag_profile_for_path("manual.pdf") == "faq"
    assert detect_rag_profile_for_path("lease.docx") == "contract"
    assert detect_rag_profile_for_path("notes.txt") == "faq"


def test_load_faq_profile_config():
    path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile="faq")
    cfg = load_rag_pipeline_config(config_path=path, config_dir=CONFIG_DIR)
    assert cfg.chunk_strategy == "seven_step"
    assert cfg.chunk_pipeline.domain == "faq"
    assert cfg.chunk_pipeline.child_target_chars == 400
    assert cfg.enable_graph_index is False
    assert cfg.ingest.mode == "ocr_only"
    assert cfg.ingest.word_to_pdf is True
    assert cfg.ingest.enable_pdf_routing is True
    assert cfg.enable_chunk_dedupe is False
    assert cfg.ingest.enable_header_footer_dedup is True
    assert cfg.retrieval.enable_router is False
    assert cfg.rewrite.enable_multi_query is False
    assert cfg.metadata.rules
    assert cfg.cleaners is not None
    assert "default" in cfg.cleaners


def test_load_contract_profile_config():
    path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile="contract")
    cfg = load_rag_pipeline_config(config_path=path, config_dir=CONFIG_DIR)
    assert cfg.chunk_strategy == "seven_step"
    assert cfg.chunk_pipeline.domain == "legal"
    assert cfg.chunk_pipeline.target_min == 100
    assert cfg.chunk_pipeline.target_max == 350
    assert cfg.enable_graph_index is False
    assert cfg.ingest.mode == "ocr_only"
    assert cfg.ingest.word_to_pdf is True
    assert cfg.enable_chunk_dedupe is False
    assert cfg.ingest.enable_pdf_routing is True
    assert cfg.retrieval.hybrid_weights == [0.5, 0.5]
    assert cfg.retrieval.enable_router is False
    assert cfg.metadata.rules
    assert cfg.cleaners is not None
    assert "legal" in str(cfg.cleaners)


def test_build_cleaner_from_rag_config_contract():
    path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile="contract")
    cfg = load_rag_pipeline_config(config_path=path, config_dir=CONFIG_DIR)
    cleaner = build_cleaner_from_rag_config(cfg)
    text = "甲方张三，电话13800138000，email@test.com"
    out = cleaner.clean(text)
    assert "[PHONE]" in out or "138" not in out
    assert "[EMAIL]" in out or "email@test.com" not in out


def test_load_rag_embedding_pipeline_config():
    path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile="faq")
    cfg = load_rag_pipeline_config(config_path=path, config_dir=CONFIG_DIR)
    assert cfg.embedding.batch_size == 32
    assert cfg.embedding.max_tokens == 512
    assert "检索" in cfg.embedding.query_instruction


def test_load_base_rag_config():
    """基座 rag.yml 与代码默认一致。"""
    path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile=None)
    cfg = load_rag_pipeline_config(config_path=path, config_dir=CONFIG_DIR)
    assert cfg.chunk_strategy == "seven_step"
    assert cfg.enable_graph_index is False
    assert cfg.ingest.mode == "ocr_only"
    assert cfg.embedding.batch_size == 32


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="未知 profile"):
        resolve_rag_pipeline_config_path(CONFIG_DIR, profile="unknown")
