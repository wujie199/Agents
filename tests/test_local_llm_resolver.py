"""本地 LLM 发现（不下载）。"""

from pathlib import Path

import pytest

from agent_platform.model.local_llm_resolver import (
    _is_complete_hf_dir,
    discover_local_llm,
)


def test_is_complete_hf_dir(tmp_path):
    d = tmp_path / "M"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert _is_complete_hf_dir(d) is False
    (d / "model.safetensors").write_bytes(b"x")
    assert _is_complete_hf_dir(d) is True


def test_discover_hf_from_tmp(tmp_path, monkeypatch):
    root = tmp_path / "llm"
    model = root / "Qwen3-0.6B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("LOCAL_LLM_ROOT", str(root))
    monkeypatch.delenv("LOCAL_LLM_MODEL_PATH", raising=False)
    disc = discover_local_llm(llm_root=root)
    assert disc is not None
    assert disc.kind == "hf"
    assert disc.model_id == "Qwen3-0.6B"


def test_discover_none_when_empty(tmp_path, monkeypatch):
    root = tmp_path / "empty"
    root.mkdir()
    monkeypatch.setenv("LOCAL_LLM_ROOT", str(root))
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:59999/v1")
    disc = discover_local_llm(llm_root=root)
    assert disc is None


def test_registry_cloud_when_no_local(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_ROOT", str(tmp_path / "none"))
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:59999/v1")
    from agent_platform.model.registry import ModelRegistry

    reg = ModelRegistry(config_path="config/models.yml")
    role = reg._roles["main_llm"]
    assert role.profile == "dashscope_main"
    assert role.fallback_chain == []
    assert reg._profiles["dashscope_main"].model_name == "qwen3.6-plus"
