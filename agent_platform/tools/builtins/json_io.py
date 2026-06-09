"""JSON 读写内置工具（替代缺失的 legacy tools 包）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


async def read_json_all_title(json_path: str | None = None, **kwargs: Any) -> dict:
    path = json_path or "data/json/read.json"
    data = _load_json(path)
    sections = data.get("sections", data)
    if isinstance(sections, dict):
        return {"titles": list(sections.keys()), "path": path}
    if isinstance(sections, list):
        return {
            "titles": [s.get("title") for s in sections if isinstance(s, dict)],
            "path": path,
        }
    return {"titles": [], "path": path}


async def read_json_context_by_title(
    title: str,
    json_path: str | None = None,
    **kwargs: Any,
) -> dict:
    path = json_path or "data/json/read.json"
    data = _load_json(path)
    sections = data.get("sections", data)
    if isinstance(sections, dict) and title in sections:
        content = sections[title]
        return {"title": title, "content": content, "path": path}
    if isinstance(sections, list):
        for item in sections:
            if isinstance(item, dict) and item.get("title") == title:
                return {"title": title, "content": item, "path": path}
    raise ValueError(f"Section not found: {title}")


async def save_result_2_json(
    result: Any = None,
    payload: Any = None,
    json_path: str | None = None,
    **kwargs: Any,
) -> dict:
    path = json_path or "data/json/save.json"
    body = result if result is not None else payload
    if body is None:
        raise ValueError("save_result_2_json requires result or payload")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(p.resolve()), "bytes": p.stat().st_size}
