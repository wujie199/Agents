"""Chat / Memory 配置精简与 deep merge 测试。"""

from pathlib import Path

from app.agents.orchestration.chat_config import (
    load_chat_config,
    load_chat_yaml_document,
    load_observability_config,
)
from agent_platform.memory.adapters.config_loader import load_memory_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO_ROOT / "config")


def test_chat_production_example_merges_base():
    raw = load_chat_yaml_document(
        f"{CONFIG_DIR}/chat.production.example.yml",
        config_dir=CONFIG_DIR,
    )
    chat = raw.get("chat") or {}
    assert chat.get("max_history_turns") == 12
    assert chat.get("rag_min_score") == 0.35


def test_load_chat_config_dev_profile():
    cfg = load_chat_config(CONFIG_DIR, profile="dev")
    assert cfg.remember_require_hitl is False
    assert cfg.rag_min_score == 0.55
    assert cfg.max_history_turns == 6


def test_load_chat_config_production_profile():
    cfg = load_chat_config(CONFIG_DIR, profile="production")
    assert cfg.max_history_turns == 12
    assert cfg.rag_min_score == 0.35
    assert cfg.remember_require_hitl is True


def test_observability_qps_from_concurrency():
    obs = load_observability_config(CONFIG_DIR)
    assert obs.max_qps_per_tenant == 100
    assert obs.audit_persist is True


def test_slim_memory_yml_defaults(monkeypatch):
    monkeypatch.delenv("MEMORY_CONFIG", raising=False)
    cfg = load_memory_config(f"{CONFIG_DIR}/memory.yml")
    assert cfg.get("store_dir") == "data/memory_dev"
    assert cfg.get("enable_session_vector_index") is True
    assert cfg.get("retention_days") == 90
