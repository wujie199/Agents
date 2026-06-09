"""Word 文档转 JSON 内置工具（轻量 stub，生产可替换为完整 OCR 流水线）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


async def read_word_2_json(
    word_path: str,
    output_path: str | None = None,
    **kwargs: Any,
) -> dict:
    path = Path(word_path)
    if not path.exists():
        raise FileNotFoundError(f"Word file not found: {word_path}")

    # Stub：仅记录路径与大小；完整解析见 document/ocr/
    payload = {
        "source": str(path.resolve()),
        "filename": path.name,
        "bytes": path.stat().st_size,
        "note": "stub: use document/ocr for full Word extraction",
        "sections": [],
    }
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["output_path"] = str(out.resolve())
    return payload
