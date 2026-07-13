# -*- coding: utf-8 -*-
"""统一 Document IR：与业务领域无关的结构化输出。"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from document.ocr.labels import ASSET_LABELS, DROP_LABELS, FORMULA_LABELS, TABLE_LABEL


def _region_content(region: dict[str, Any]) -> dict[str, Any]:
    label = region.get("label", "")
    if label == TABLE_LABEL:
        tr = region.get("table_result") or {}
        return {
            "type": "table",
            "html": tr.get("pred_html") or region.get("text") or "",
            "table_type": tr.get("table_type"),
        }
    if label in FORMULA_LABELS:
        return {
            "type": "formula",
            "latex": region.get("text") or "",
            "rec_score": region.get("rec_score"),
        }
    if label in ASSET_LABELS:
        return {"type": "asset", "label": label}
    if region.get("text"):
        return {
            "type": "text",
            "text": region.get("text") or "",
            "rec_score": region.get("rec_score"),
        }
    return {"type": "unknown", "label": label}


def normalize_region(region: dict[str, Any]) -> dict[str, Any]:
    """在原有字段上附加 content，保持向后兼容。"""
    out = dict(region)
    out["content"] = _region_content(region)
    return out


def normalize_page(page: dict[str, Any]) -> dict[str, Any]:
    out = dict(page)
    out["regions"] = [normalize_region(r) for r in page.get("regions") or []]
    return out


def _region_sort_key(region: dict[str, Any]) -> tuple:
    order = region.get("order")
    coord = region.get("coordinate") or [0, 0, 0, 0]
    if order is None:
        return (1, coord[1] if len(coord) > 1 else 0, coord[0] if coord else 0)
    return (0, order)


def build_document_ir(
    *,
    source: Path,
    pages: list[dict[str, Any]],
    pipeline_version: str,
    config_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_pages = [normalize_page(p) for p in pages]
    page_count = len(normalized_pages)
    qc_pass = 0
    qc_fail = 0
    recovered_by_retry = 0
    for p in normalized_pages:
        qc = p.get("qc") or {}
        status = qc.get("status")
        attempt = int(p.get("attempt") or 0)
        if status == "pass":
            qc_pass += 1
            if attempt > 0:
                recovered_by_retry += 1
        elif status == "fail":
            qc_fail += 1

    try:
        src = str(source.resolve())
    except OSError:
        src = str(source)

    return {
        "schema": "document_ir/1.0",
        "pipeline_version": pipeline_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": src,
        "source_name": source.name,
        "page_count": page_count,
        "qc_summary": {
            "pass_pages": qc_pass,
            "fail_pages": qc_fail,
            "recovered_by_retry": recovered_by_retry,
        },
        "config": config_summary or {},
        "pages": normalized_pages,
    }


def region_to_markdown(region: dict[str, Any]) -> str:
    label = region.get("label", "")
    if label in DROP_LABELS:
        return ""
    content = region.get("content") or _region_content(region)
    ctype = content.get("type")

    if ctype == "table":
        html = content.get("html") or ""
        if html:
            return f"\n**[table]**\n\n{html}\n"
        return ""

    if ctype == "formula":
        latex = content.get("latex") or ""
        if latex:
            return f"\n**[{label}]** $${latex}$$\n"
        return ""

    if ctype == "text":
        text = content.get("text") or ""
        if text:
            return f"\n**[{label}]** {text}\n"
        return ""

    return ""


def document_to_markdown(document: dict[str, Any], *, title: str | None = None) -> str:
    name = title or document.get("source_name") or "document"
    lines = [f"# OCR: {name}\n"]
    for page in document.get("pages") or []:
        lines.append(f"\n## 第 {page.get('page_index', 0) + 1} 页\n")
        qc = page.get("qc", {})
        if qc.get("status") and qc["status"] != "pass":
            lines.append(f"\n> QC: {qc['status']} ({qc.get('issue_count', 0)} issues)\n")
        regions = sorted(page.get("regions") or [], key=_region_sort_key)
        for r in regions:
            lines.append(region_to_markdown(r))
    return "".join(lines)


def source_fingerprint(path: Path) -> str:
    """源文件哈希，可用于去重缓存（调用方按需使用）。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
