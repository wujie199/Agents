# -*- coding: utf-8 -*-
"""Memory YAML 加载：分段 config/memory.yml → 扁平 dict（兼容现有 mem_cfg.get）。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

MEMORY_SECTIONS = (
    "l1",
    "l2",
    "l0",
    "archive",
    "vector",
    "skills",
    "l4",
    "cold_archive",
)

MEMORY_PROFILES = frozenset({"vector", "cold", "skills", "l4_http"})

_FLAT_MARKERS = frozenset(
    {
        "hot_memory_max_chars",
        "archive_backend",
        "enable_session_vector_index",
        "store_dir",
    }
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_flat_memory_raw(raw: dict[str, Any]) -> bool:
    if not raw:
        return False
    if any(section in raw for section in MEMORY_SECTIONS):
        return False
    return any(key in raw for key in _FLAT_MARKERS)


def flatten_memory_yaml(raw: dict[str, Any]) -> dict[str, Any]:
    """将分段 memory.yml 展平为 legacy 键值 dict。"""
    if _is_flat_memory_raw(raw):
        return {k: v for k, v in raw.items() if k != "profiles"}

    flat: dict[str, Any] = {}
    for section in MEMORY_SECTIONS:
        block = raw.get(section)
        if isinstance(block, dict):
            flat.update(block)
    for key, value in raw.items():
        if key not in MEMORY_SECTIONS and key != "profiles":
            flat[key] = value
    return flat


def apply_memory_profile(
    flat: dict[str, Any],
    raw: dict[str, Any],
    profile: Optional[str],
) -> dict[str, Any]:
    if not profile:
        return flat
    key = profile.strip()
    profiles = raw.get("profiles") or {}
    if key not in profiles:
        if key in MEMORY_PROFILES:
            raise ValueError(
                f"未知 MEMORY_PROFILE={key!r}，可选: {', '.join(sorted(MEMORY_PROFILES))}"
            )
        return flat
    overlay = profiles[key]
    if not isinstance(overlay, dict):
        return flat
    merged = dict(flat)
    merged.update(overlay)
    return merged


def memory_base_path(config_dir: str | Path = "config") -> Path:
    return Path(config_dir) / "memory.yml"


def resolve_memory_config_path(
    config_dir: str = "config",
    *,
    profile: str = "dev",
) -> str:
    """解析记忆配置文件路径（尊重 MEMORY_CONFIG / production 默认）。"""
    env_path = os.environ.get("MEMORY_CONFIG")
    if env_path:
        return env_path
    base = Path(config_dir)
    if profile == "production":
        for name in ("memory.production.yml", "memory.production.example.yml"):
            candidate = base / name
            if candidate.is_file():
                return str(candidate)
    legacy = base / "memory.dev-vector.example.yml"
    rag_style = base / "memory.yml"
    if rag_style.is_file():
        return str(rag_style)
    if legacy.is_file():
        return str(legacy)
    return str(rag_style)


def load_memory_yaml_document(
    config_path: str | Path,
    *,
    config_dir: str = "config",
    memory_profile: Optional[str] = None,
) -> dict[str, Any]:
    """加载 YAML 并展平；MEMORY_PROFILE 仅对 config/memory.yml 生效。"""
    path = Path(config_path)
    raw = _read_yaml(path)

    # 若指向已删除的 dev example，尝试从 base + profile 恢复
    legacy_profile_map = {
        "memory.dev-vector.example.yml": "vector",
        "memory.dev-cold.example.yml": "cold",
        "memory.dev-skills.example.yml": "skills",
        "memory.dev-l4-http.example.yml": "l4_http",
    }
    if not raw and path.name in legacy_profile_map:
        logger.warning(
            "配置文件 %s 已废弃，改用 MEMORY_PROFILE=%s + config/memory.yml",
            path.name,
            legacy_profile_map[path.name],
        )
        base = memory_base_path(config_dir)
        raw = _read_yaml(base)
        memory_profile = memory_profile or legacy_profile_map[path.name]

    flat = flatten_memory_yaml(raw)
    profile_key = memory_profile or os.environ.get("MEMORY_PROFILE")
    if profile_key:
        profile_source = _read_yaml(memory_base_path(config_dir))
        if not (profile_source.get("profiles")):
            profile_source = _read_yaml(memory_base_path("config"))
        if not profile_source.get("profiles"):
            profile_source = raw
        flat = apply_memory_profile(flat, profile_source, profile_key)
    return flat
