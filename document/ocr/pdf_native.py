# -*- coding: utf-8 -*-
"""原生 PDF 文本层抽取（带 bbox），输出与 document_ir 兼容的页结构。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TextSpan:
    text: str
    bbox: Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    page_index: int


def _group_spans_into_lines(spans: List[TextSpan], *, y_tolerance: float = 4.0) -> List[List[TextSpan]]:
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: (s.bbox[1], s.bbox[0]))
    lines: List[List[TextSpan]] = []
    current: List[TextSpan] = []
    current_y: Optional[float] = None

    for sp in sorted_spans:
        y = sp.bbox[1]
        if current_y is None or abs(y - current_y) <= y_tolerance:
            current.append(sp)
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            if current:
                current.sort(key=lambda s: s.bbox[0])
                lines.append(current)
            current = [sp]
            current_y = y
    if current:
        current.sort(key=lambda s: s.bbox[0])
        lines.append(current)
    return lines


def _line_to_region(line: List[TextSpan], page_index: int, order: int) -> Dict[str, Any]:
    text = "".join(s.text for s in line).strip()
    if not text:
        return {}
    xs = [s.bbox[0] for s in line] + [s.bbox[2] for s in line]
    ys = [s.bbox[1] for s in line] + [s.bbox[3] for s in line]
    coord = [min(xs), min(ys), max(xs), max(ys)]
    avg_height = sum(s.bbox[3] - s.bbox[1] for s in line) / max(1, len(line))
    label = "paragraph_title" if avg_height > 14 and len(text) < 80 else "text"
    if len(text) < 40 and not text.endswith(("。", ".", "!", "?")):
        if avg_height > 12:
            label = "paragraph_title"
    return {
        "box_index": order,
        "order": order,
        "label": label,
        "layout_score": 1.0,
        "coordinate": coord,
        "text": text,
        "rec_score": 1.0,
        "content": {"type": "text", "text": text, "rec_score": 1.0},
        "source": "pdf_text_layer",
    }


def _extract_native_page_from_doc(doc: object, page_index: int) -> Dict[str, Any]:
    page = doc[page_index]
    width, height = page.get_size()
    spans: List[TextSpan] = []
    textpage = None

    try:
        textpage = page.get_textpage()
        n = textpage.count_chars()
        for i in range(n):
            ch = textpage.get_text_range(i, 1)
            if not ch or ch.isspace() and ch != " ":
                continue
            box = textpage.get_charbox(i)
            if not box:
                continue
            left, bottom, right, top = box
            ymin = float(height - top)
            ymax = float(height - bottom)
            spans.append(
                TextSpan(
                    text=ch,
                    bbox=(float(left), ymin, float(right), ymax),
                    page_index=page_index,
                )
            )
    except Exception:
        return {
            "page_index": page_index,
            "image": "",
            "regions": [],
            "qc": {
                "status": "fail",
                "issues": [{"code": "native_extract_failed", "message": "文本层抽取失败"}],
            },
            "route": "native",
        }
    finally:
        if textpage is not None:
            textpage.close()

    lines = _group_spans_into_lines(spans)
    regions: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        region = _line_to_region(line, page_index, idx)
        if region:
            regions.append(region)

    full_text = "\n".join(r["text"] for r in regions if r.get("text"))
    return {
        "page_index": page_index,
        "image": "",
        "regions": regions,
        "full_text": full_text,
        "route": "native",
        "qc": {"status": "pass", "issues": [], "issue_count": 0},
        "page_size": [float(width), float(height)],
    }


def extract_native_page(pdf_path: Path, page_index: int) -> Dict[str, Any]:
    """从 PDF 文本层抽取单页，返回 normalize_page 兼容结构。"""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return _extract_native_page_from_doc(doc, page_index)
    finally:
        doc.close()


def extract_native_pages(
    pdf_path: Path,
    page_indices: List[int],
) -> Dict[int, Dict[str, Any]]:
    """单次打开 PDF，批量抽取 native 页（避免重复 open/close）。"""
    if not page_indices:
        return {}

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return {i: _extract_native_page_from_doc(doc, i) for i in page_indices}
    finally:
        doc.close()


def extract_native_document_pages(
    pdf_path: Path,
    page_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        n = len(doc)
        indices = page_indices if page_indices is not None else list(range(n))
        return [_extract_native_page_from_doc(doc, i) for i in indices]
    finally:
        doc.close()
