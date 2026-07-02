"""Memory YAML 加载与 profile 测试。"""

from __future__ import annotations

from agent_platform.memory.adapters.config_loader import load_memory_config
from agent_platform.memory.adapters.memory_yaml import (
    flatten_memory_yaml,
    load_memory_yaml_document,
)


def test_flatten_sectioned_memory_yml():
    raw = {
        "l1": {"store_dir": "data/x", "hot_memory_max_chars": 1000},
        "archive": {"archive_backend": "sqlite"},
        "profiles": {"vector": {"enable_cold_archive": False}},
    }
    flat = flatten_memory_yaml(raw)
    assert flat["store_dir"] == "data/x"
    assert flat["archive_backend"] == "sqlite"
    assert "profiles" not in flat


def test_memory_profile_vector(monkeypatch):
    monkeypatch.setenv("MEMORY_PROFILE", "vector")
    cfg = load_memory_config("config/memory.yml")
    assert cfg.get("enable_session_vector_index") is True
    assert cfg.get("enable_cold_archive") is False
    assert cfg.get("session_vector_dir") == "data/session_vectors_dev"


def test_memory_yaml_l0_section():
    cfg = load_memory_config("config/memory.yml")
    assert cfg.get("l0_context_compress_enabled") is True
    assert cfg.get("context_compress_threshold") == 0.50
    assert cfg.get("compress_target_ratio") == 0.20


def test_memory_profile_l4_http(monkeypatch):
    monkeypatch.setenv("MEMORY_PROFILE", "l4_http")
    cfg = load_memory_config("config/memory.yml")
    assert cfg.get("external_profiles_backend") == "http"
    assert "8765" in str(cfg.get("external_profiles_http_url"))


def test_production_example_nested():
    flat = load_memory_yaml_document("config/memory.production.example.yml")
    assert flat.get("archive_backend") == "postgresql"
    assert flat.get("l1_store_backend") == "relational"


def test_legacy_memory_config_env_fallback(monkeypatch):
    monkeypatch.setenv("MEMORY_CONFIG", "config/memory.dev-vector.example.yml")
    cfg = load_memory_config()
    assert cfg.get("session_vector_dir") == "data/session_vectors_dev"
    assert cfg.get("enable_cold_archive") is False
