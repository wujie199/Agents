# -*- coding: utf-8 -*-
"""RAG YAML 加载：统一 config/rag.yml 及 profile 解析。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

RAG_PROFILES = frozenset({"faq", "contract"})
# 兼容旧名
RAG_PIPELINE_PROFILES = RAG_PROFILES

_LEGACY_PROFILE_PREFIX = "rag_pipeline."
_NEW_PROFILE_PREFIX = "rag."


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rag_base_path(config_dir: str | Path = "config") -> Path:
    return Path(config_dir) / "rag.yml"


def load_rag_base_yaml(config_dir: str | Path = "config") -> dict[str, Any]:
    """加载 config/rag.yml（含 metadata.rules、scenarios、eval 等共享段）。"""
    return _read_yaml(rag_base_path(config_dir))


def _resolve_profile_file(config_dir: Path, profile: str) -> Path:
    key = profile.lower().strip()
    if key not in RAG_PROFILES:
        raise ValueError(
            f"未知 profile: {profile!r}，可选: {', '.join(sorted(RAG_PROFILES))}"
        )
    for name in (f"rag.{key}.yml", f"rag_pipeline.{key}.yml"):
        path = config_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"profile 配置文件不存在: rag.{key}.yml")


def resolve_rag_config_path(
    config_dir: str = "config",
    profile: Optional[str] = None,
) -> str:
    """解析 RAG YAML 路径：profile > RAG_CONFIG / RAG_PIPELINE_CONFIG > 默认 rag.yml。"""
    base = Path(config_dir)
    if profile:
        return str(_resolve_profile_file(base, profile))

    for env_key in ("RAG_CONFIG", "RAG_PIPELINE_CONFIG"):
        env_path = os.environ.get(env_key)
        if env_path:
            return env_path

    rag_path = base / "rag.yml"
    if rag_path.is_file():
        return str(rag_path)
    legacy = base / "rag_pipeline.yml"
    if legacy.is_file():
        logger.warning("使用已废弃的 config/rag_pipeline.yml，请迁移至 config/rag.yml")
        return str(legacy)
    return str(rag_path)


def resolve_rag_pipeline_config_path(
    config_dir: str = "config",
    profile: Optional[str] = None,
) -> str:
    """兼容旧 API。"""
    return resolve_rag_config_path(config_dir, profile)


def merge_metadata_from_base(
    metadata_raw: dict[str, Any],
    base_raw: dict[str, Any],
) -> dict[str, Any]:
    """profile 文件未内嵌 rules 时，从 rag.yml 继承 metadata.rules。"""
    merged = dict(metadata_raw)
    base_meta = dict(base_raw.get("metadata") or {})
    if not merged.get("rules") and base_meta.get("rules"):
        merged["rules"] = base_meta["rules"]
    if not merged.get("extension_tags") and base_meta.get("extension_tags"):
        merged["extension_tags"] = base_meta["extension_tags"]
    return merged


def deep_merge_rag_config(
    base: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    """rag.yml 基座 + profile 增量：嵌套 dict 递归合并，标量/列表以 overlay 为准。"""
    result = dict(base)
    for key, overlay_val in overlay.items():
        base_val = result.get(key)
        if (
            isinstance(overlay_val, dict)
            and isinstance(base_val, dict)
        ):
            result[key] = deep_merge_rag_config(base_val, overlay_val)
        else:
            result[key] = overlay_val
    return result


def _is_rag_base_document(path: Path) -> bool:
    return path.name in ("rag.yml", "rag_pipeline.yml")


def load_rag_yaml_document(
    config_path: str | Path,
    *,
    config_dir: str = "config",
) -> dict[str, Any]:
    """加载 RAG 配置：profile/示例外层文件与 config/rag.yml 深合并。"""
    path = Path(config_path)
    raw = _read_yaml(path)
    base_raw = load_rag_base_yaml(config_dir)
    if _is_rag_base_document(path) or not base_raw:
        return raw
    return deep_merge_rag_config(base_raw, raw)


def load_rag_eval_section(config_dir: str | Path = "config") -> dict[str, Any]:
    """从 rag.yml 读取 eval 段；兼容旧 rag_eval.yml。"""
    base = load_rag_base_yaml(config_dir)
    eval_raw = dict(base.get("eval") or {})
    if eval_raw:
        return eval_raw

    legacy_path = Path(config_dir) / "rag_eval.yml"
    if legacy_path.is_file():
        logger.warning("使用已废弃的 config/rag_eval.yml，请迁移至 config/rag.yml 的 eval 段")
        legacy = _read_yaml(legacy_path)
        return {
            "generation": legacy.get("generation") or {},
            "judge": legacy.get("judge") or {},
            "roles": legacy.get("roles") or {},
            "metrics": legacy.get("metrics") or {},
            "output": legacy.get("output") or {},
        }
    return {}


def load_rag_scenarios_section(config_dir: str | Path = "config") -> dict[str, Any]:
    """从 rag.yml 读取 scenarios；兼容旧 scenarios.yml。"""
    base = load_rag_base_yaml(config_dir)
    scenarios_raw = dict(base.get("scenarios") or {})
    if scenarios_raw:
        items = scenarios_raw.get("items") or scenarios_raw.get("scenarios") or {}
        return {
            "default_scenario": scenarios_raw.get("default_scenario"),
            "scenarios": items,
        }

    legacy_path = Path(config_dir) / "scenarios.yml"
    if legacy_path.is_file():
        logger.warning("使用已废弃的 config/scenarios.yml，请迁移至 config/rag.yml 的 scenarios 段")
        legacy = _read_yaml(legacy_path)
        if "scenarios" not in legacy:
            legacy["scenarios"] = {}
        return legacy
    return {"scenarios": {}, "default_scenario": None}


def load_rag_chroma_section(config_dir: str | Path = "config") -> dict[str, Any]:
    """从 rag.yml storage 段读取 Chroma 配置；兼容旧 chroma.yml。"""
    base = load_rag_base_yaml(config_dir)
    storage = dict(base.get("storage") or {})
    chroma = dict(storage.get("chroma") or {})
    legacy_storage = dict(storage.get("legacy") or {})
    merged = {**legacy_storage, **chroma}

    legacy_path = Path(config_dir) / "chroma.yml"
    if legacy_path.is_file() and not merged:
        logger.warning("使用已废弃的 config/chroma.yml，请迁移至 config/rag.yml 的 storage 段")
        merged = _read_yaml(legacy_path)
    return merged
