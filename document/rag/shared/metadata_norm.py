"""元数据规范化：字段别名映射 + 日期/标签值归一化。"""

import re
from datetime import datetime
from typing import Any, Dict

try:
    from dateutil.parser import parse as _date_parse
except ImportError:
    _date_parse = None

# 常用 metadata 字段别名映射
_METADATA_FIELD_ALIASES = {
    "author": ["author", "作者", "writer", "written_by", "creator"],
    "title": ["title", "标题", "name"],
    "source": ["source", "来源", "origin"],
    "source_id": ["source_id", "sourceid", "source-id", "id", "doc_id", "document_id"],
    "created_at": ["created_at", "created", "creation_date", "date", "date_created", "创建时间"],
    "updated_at": ["updated_at", "updated", "modified", "modified_at", "date_modified", "更新时间"],
    "tags": ["tags", "tag", "keywords", "keyword", "labels", "标签"],
    "summary": ["summary", "description", "desc", "摘要"],
}

_ALIAS_LOOKUP = {alias: canonical for canonical, aliases in _METADATA_FIELD_ALIASES.items() for alias in aliases}


def _normalize_metadata_key(key: Any) -> str:
    if key is None:
        return ""
    s = str(key).strip().lower().replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^\w\u4e00-\u9fff_]", "", s, flags=re.UNICODE)
    return _ALIAS_LOOKUP.get(s, s)


def _normalize_date_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    s = str(value).strip()
    if not s:
        return ""
    if _date_parse:
        try:
            return _date_parse(s, fuzzy=True).isoformat()
        except (ValueError, OverflowError):
            pass
    match = re.search(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", s)
    if match:
        y, m, d = match.groups()
        try:
            return datetime(int(y), int(m), int(d)).isoformat()
        except ValueError:
            return s
    return s


def normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metadata keys and values."""
    from document.rag.shared.data_cleaner import clean_text

    if not isinstance(meta, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        key = _normalize_metadata_key(k)
        if not key:
            continue
        if key in ("created_at", "updated_at", "date"):
            out[key] = _normalize_date_value(v)
            continue
        if key == "tags":
            if isinstance(v, str):
                tags = [t.strip() for t in re.split(r"[;,\|]+", v) if t.strip()]
                out[key] = tags
                continue
            if isinstance(v, (list, tuple, set)):
                out[key] = [str(t).strip() for t in v if t is not None]
                continue
        if isinstance(v, str):
            out[key] = clean_text(v)
        else:
            out[key] = v
    return out
