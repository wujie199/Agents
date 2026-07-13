"""OCR 摄取文本：从 document_ir 重建正文，轻量清洗（保留 FAQ/Markdown 结构）。"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

from document.ocr.labels import DROP_LABELS
from document.rag.shared.data_cleaner import postprocess_ocr


def _region_plain_text(region: Dict[str, Any]) -> str:
    label = str(region.get("label") or "")
    if label in DROP_LABELS:
        return ""
    content = region.get("content") or {}
    ctype = content.get("type")
    if ctype == "table":
        return (content.get("html") or region.get("text") or "").strip()
    if ctype == "formula":
        latex = content.get("latex") or region.get("text") or ""
        return f"$${latex}$$" if latex else ""
    if ctype == "text":
        return (content.get("text") or region.get("text") or "").strip()
    return str(region.get("text") or "").strip()


def page_plain_text_from_ir(page: Dict[str, Any]) -> str:
    """单页 IR → 换行拼接正文（无页眉前缀）。"""
    parts: List[str] = []
    for region in page.get("regions") or []:
        text = _region_plain_text(region)
        if text:
            parts.append(text)
    return "\n".join(parts)


def rebuild_content_from_document_ir(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """从 metadata.document_ir 重建与 OCR region 一致的全文。"""
    if not metadata:
        return None
    ir = metadata.get("document_ir")
    if not isinstance(ir, dict):
        return None
    pages = ir.get("pages") or []
    if not pages:
        return None
    parts: List[str] = []
    for page in pages:
        pn = int(page.get("page_index", 0)) + 1
        body = page_plain_text_from_ir(page)
        if body:
            parts.append(f"=== 第 {pn} 页 ===\n{body}")
    return "\n\n".join(parts) if parts else None


def apply_ocr_light_cleaning(text: str) -> str:
    """OCR 专用轻量清洗：去控制字符/OCR 噪声，保留 #、**、小数点与换行。"""
    if not text:
        return text
    s = postprocess_ocr(text, preserve_layout=True, preserve_tables=True)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    return s.strip()


def sync_pages_content_from_ir(ingest_pages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    """将 ingest pages[].content 与 document_ir 对齐。"""
    ir = metadata.get("document_ir")
    if not isinstance(ir, dict):
        return
    ir_by_num: Dict[int, str] = {}
    for page in ir.get("pages") or []:
        pn = int(page.get("page_index", 0)) + 1
        ir_by_num[pn] = page_plain_text_from_ir(page)
    for page in ingest_pages:
        pn = int(page.get("page_num") or (int(page.get("page_index", 0)) + 1))
        if pn in ir_by_num:
            page["content"] = ir_by_num[pn]
            page["char_count"] = len(page["content"])
