"""RAG profile 配置选择与加载测试。"""

from pathlib import Path

import pytest

from document.rag.config import (
    detect_rag_profile_for_path,
    load_rag_pipeline_config,
    resolve_rag_pipeline_config_path,
)
from document.rag.components.cleaner.factory import build_cleaner_from_rag_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO_ROOT / "config")


def test_resolve_profile_paths():
    faq_path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile="faq")
    assert faq_path.endswith("rag_pipeline.faq.yml")
    contract_path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile="contract")
    assert contract_path.endswith("rag_pipeline.contract.yml")


def test_detect_rag_profile_for_path():
    assert detect_rag_profile_for_path("manual.pdf") == "faq"
    assert detect_rag_profile_for_path("lease.docx") == "contract"
    assert detect_rag_profile_for_path("notes.txt") == "faq"


def test_load_faq_profile_config():
    path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile="faq")
    cfg = load_rag_pipeline_config(config_path=path, config_dir=CONFIG_DIR)
    assert cfg.chunk_strategy == "faq"
    assert cfg.enable_chunk_dedupe is True
    assert cfg.ingest.enable_header_footer_dedup is True
    assert cfg.cleaners is not None
    assert "default" in cfg.cleaners


def test_load_contract_profile_config():
    path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile="contract")
    cfg = load_rag_pipeline_config(config_path=path, config_dir=CONFIG_DIR)
    assert cfg.chunk_strategy == "article"
    assert cfg.ingest.mode == "structured"
    assert cfg.ingest.word_to_pdf is False
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


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="未知 profile"):
        resolve_rag_pipeline_config_path(CONFIG_DIR, profile="unknown")
