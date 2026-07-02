"""文档标签过滤：单库多场景检索。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

MatchMode = str  # "any" | "all"


def parse_tags_value(raw: Any) -> List[str]:
    """从 metadata.tags / tags_csv 解析标签列表。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [t.strip() for t in re.split(r"[,;|]+", raw) if t.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(t).strip() for t in raw if t is not None and str(t).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def metadata_matches_tags(
    metadata: Optional[Dict[str, Any]],
    required_tags: Sequence[str],
    *,
    match: MatchMode = "any",
) -> bool:
    """判断文档 metadata 是否满足标签条件。"""
    if not required_tags:
        return True
    meta = metadata or {}
    doc_tags = set(parse_tags_value(meta.get("tags")) or parse_tags_value(meta.get("tags_csv")))
    if not doc_tags:
        return False
    required = [t.strip() for t in required_tags if t and str(t).strip()]
    if not required:
        return True
    if match == "all":
        return all(t in doc_tags for t in required)
    return any(t in doc_tags for t in required)


def chroma_safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma 仅支持标量 metadata；列表字段序列化为逗号分隔字符串。"""
    safe: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            joined = ",".join(str(v) for v in value if v is not None and str(v).strip())
            safe[key] = joined
            if key == "tags":
                safe["tags_csv"] = joined
        elif isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def merge_tags_into_metadata(
    metadata: Dict[str, Any],
    extra_tags: Optional[Sequence[str]],
) -> Dict[str, Any]:
    """合并手动标签到 metadata.tags（去重保序）。"""
    if not extra_tags:
        return metadata
    meta = dict(metadata)
    existing = parse_tags_value(meta.get("tags"))
    seen = set(existing)
    merged = list(existing)
    for tag in extra_tags:
        t = str(tag).strip()
        if t and t not in seen:
            seen.add(t)
            merged.append(t)
    meta["tags"] = merged
    meta["categories"] = merged
    return meta


from document.rag.config.rag_yaml import load_rag_scenarios_section


def load_scenario_config(config_dir: str = "config") -> Dict[str, Any]:
    """加载 config/rag.yml 的 scenarios 段。"""
    return load_rag_scenarios_section(config_dir)


def resolve_scenario_tags(
    scenario: Optional[str],
    *,
    config_dir: str = "config",
    explicit_tags: Optional[Sequence[str]] = None,
    tag_match: Optional[str] = None,
) -> tuple[List[str], str]:
    """
    解析场景或直接指定的标签。

    Returns:
        (tags, match_mode)
    """
    cfg = load_scenario_config(config_dir)
    scenarios = cfg.get("scenarios") or {}

    tags: List[str] = []
    match = tag_match or "any"

    if scenario:
        entry = scenarios.get(scenario)
        if entry is None:
            raise KeyError(f"未知场景 {scenario!r}，请检查 config/rag.yml 的 scenarios 段")
        if isinstance(entry, dict):
            tags.extend(parse_tags_value(entry.get("tags")))
            match = tag_match or str(entry.get("tag_match") or "any")
        elif isinstance(entry, list):
            tags.extend(parse_tags_value(entry))
        else:
            tags.extend(parse_tags_value(entry))

    if explicit_tags:
        for t in explicit_tags:
            t = str(t).strip()
            if t and t not in tags:
                tags.append(t)

    return tags, match
