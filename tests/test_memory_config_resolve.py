"""记忆配置路径解析。"""

from __future__ import annotations

import os
from pathlib import Path

from agent_platform.memory.adapters.config_loader import (
    ensure_memory_config_env,
    resolve_memory_config_path,
)


def test_resolve_production_example(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "memory.production.example.yml").write_text(
        "archive_backend: postgresql\n", encoding="utf-8"
    )
    path = resolve_memory_config_path(str(cfg_dir), profile="production")
    assert path.endswith("memory.production.example.yml")


def test_memory_config_env_setdefault(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORY_CONFIG", raising=False)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    prod = cfg_dir / "memory.production.example.yml"
    prod.write_text("store_dir: x\n", encoding="utf-8")
    resolved = ensure_memory_config_env(str(cfg_dir), profile="production")
    monkeypatch.setenv("MEMORY_CONFIG", resolved)
    assert os.environ.get("MEMORY_CONFIG") == resolved
    assert resolved == str(prod)
