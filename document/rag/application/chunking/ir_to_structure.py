"""从 OCR document_ir 构建七步分块 Step1 结构单元（P1）。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from document.ocr.labels import DROP_LABELS, TABLE_LABEL, TITLE_LABELS
from document.rag.application.chunking.models import StructuralUnit

_NUMBERED_FAQ_HEAD = re.compile(r"^\d{1,3}[\.．、)]\s*(\*\*)?")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_CN_SECTION = re.compile(r"^[#\s]*[一二三四五六七八九十百]+[、．.]")


def _html_table_to_markdown(html: str) -> str:
    if not html or "<table" not in html.lower():
        return html
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.I | re.S)
    lines: List[str] = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.I | re.S)
        if not cells:
            continue
        text_cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        lines.append("| " + " | ".join(text_cells) + " |")
    if len(lines) >= 2:
        ncol = lines[0].count("|") - 1
        sep = "| " + " | ".join(["---"] * max(1, ncol)) + " |"
        lines.insert(1, sep)
    return "\n".join(lines)


def _region_text(region: Dict[str, Any]) -> str:
    content = region.get("content") or {}
    ctype = content.get("type")
    if ctype == "table":
        html = content.get("html") or region.get("text") or ""
        return _html_table_to_markdown(html) or html
    if ctype == "formula":
        latex = content.get("latex") or region.get("text") or ""
        return f"$${latex}$$" if latex else ""
    if ctype == "text":
        return content.get("text") or region.get("text") or ""
    return region.get("text") or ""


def _is_numbered_faq_line(text: str) -> bool:
    """版面误标为 title 的 numbered FAQ 问句。"""
    t = text.strip()
    if _NUMBERED_FAQ_HEAD.match(t):
        return True
    if re.match(r"^\d{1,3}\.", t) and ("？" in t or "?" in t):
        return True
    return False


def _heading_level(text: str, label: str) -> int:
    m = _MD_HEADING.match(text.strip())
    if m:
        return len(m.group(1))
    if label == "doc_title":
        return 1
    if _CN_SECTION.match(text.strip()):
        return 2
    if text.strip().startswith("###"):
        return 3
    if text.strip().startswith("##"):
        return 2
    if text.strip().startswith("#"):
        return 1
    return 2


def _should_emit_heading(label: str, text: str) -> bool:
    if label not in TITLE_LABELS:
        return False
    if _is_numbered_faq_line(text):
        return False
    return True


def _region_unit_type(label: str, text: str = "") -> str:
    if label == TABLE_LABEL:
        return "table"
    if _should_emit_heading(label, text):
        return "heading"
    if label in ("list", "catalogue"):
        return "list"
    if label in ("code", "algorithm"):
        return "code"
    return "paragraph"


def units_from_document_ir(
    document_ir: Dict[str, Any],
    *,
    default_heading: str = "",
) -> List[StructuralUnit]:
    """将 document_ir 转为 Step1 StructuralUnit 列表，保留标题路径与页码。"""
    units: List[StructuralUnit] = []
    position = 0
    heading_stack: List[str] = []

    for page in document_ir.get("pages") or []:
        page_index = int(page.get("page_index", 0))
        page_num = page_index + 1

        for region in page.get("regions") or []:
            label = str(region.get("label") or "text")
            if label in DROP_LABELS:
                continue
            text = _region_text(region).strip()
            if not text:
                continue

            unit_type = _region_unit_type(label, text)
            if unit_type == "heading":
                level = _heading_level(text, label)
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(text[:120])
                path = " > ".join(heading_stack)
                units.append(
                    StructuralUnit(
                        unit_type="heading",
                        content=text,
                        heading_path=path,
                        position=position,
                        metadata={
                            "page": page_num,
                            "block_type": label,
                            "bbox": region.get("coordinate"),
                            "source": "document_ir",
                        },
                    )
                )
                position += 1
                continue

            path = " > ".join(heading_stack) if heading_stack else default_heading
            coord = region.get("coordinate") or []
            units.append(
                StructuralUnit(
                    unit_type=unit_type,
                    content=text,
                    heading_path=path,
                    position=position,
                    metadata={
                        "page": page_num,
                        "block_type": label,
                        "bbox": coord,
                        "source": "document_ir",
                    },
                )
            )
            position += 1

    return units


def get_document_ir_from_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metadata:
        return None
    ir = metadata.get("document_ir")
    if isinstance(ir, dict) and ir.get("pages"):
        return ir
    return None
