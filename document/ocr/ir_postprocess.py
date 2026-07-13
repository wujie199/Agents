# -*- coding: utf-8 -*-
"""Document IR 后处理：跨页表格合并、丢弃块过滤、阅读顺序。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from document.ocr.labels import DROP_LABELS, TABLE_LABEL, TITLE_LABELS
from document.ocr.reading_order import sort_boxes_reading_order


def filter_dropped_regions(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """移除 header/footer/页码/水印等版面层黑名单块。"""
    return [r for r in regions if str(r.get("label") or "") not in DROP_LABELS]


def _table_html(region: Dict[str, Any]) -> str:
    tr = region.get("table_result") or {}
    content = region.get("content") or {}
    return (
        tr.get("pred_html")
        or content.get("html")
        or region.get("text")
        or ""
    ).strip()


def _table_header_signature(html: str) -> str:
    if not html:
        return ""
    first_row = re.search(r"<tr[^>]*>(.*?)</tr>", html, re.I | re.S)
    if not first_row:
        return html[:120]
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", first_row.group(1), re.I | re.S)
    text = "|".join(re.sub(r"<[^>]+>", "", c).strip() for c in cells)
    return re.sub(r"\s+", " ", text)[:160]


def merge_cross_page_tables(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    P2：跨页表格合并。
    若相邻页末表与次页首表表头签名一致，合并 HTML 行并删除重复表头。
    """
    pages = document.get("pages") or []
    if len(pages) < 2:
        return document

    merged_pages: List[Dict[str, Any]] = []
    carry_table: Optional[Dict[str, Any]] = None

    for page in pages:
        page = dict(page)
        regions = list(page.get("regions") or [])
        if carry_table is not None:
            if regions and regions[0].get("label") == TABLE_LABEL:
                head_html = _table_html(regions[0])
                if _table_header_signature(head_html) == carry_table.get("_header_sig"):
                    combined = _merge_table_html(carry_table["html"], head_html)
                    merged_region = dict(carry_table["region"])
                    merged_region["text"] = combined
                    if merged_region.get("table_result"):
                        merged_region["table_result"] = dict(merged_region["table_result"])
                        merged_region["table_result"]["pred_html"] = combined
                    content = dict(merged_region.get("content") or {})
                    content["html"] = combined
                    content["type"] = "table"
                    merged_region["content"] = content
                    if merged_pages:
                        merged_pages[-1]["regions"] = list(merged_pages[-1].get("regions") or []) + [
                            merged_region
                        ]
                    else:
                        regions[0] = merged_region
                    regions = regions[1:]
                    carry_table = None
                else:
                    if merged_pages:
                        merged_pages[-1]["regions"] = list(merged_pages[-1].get("regions") or []) + [
                            carry_table["region"]
                        ]
                    carry_table = None
            else:
                if merged_pages:
                    merged_pages[-1]["regions"] = list(merged_pages[-1].get("regions") or []) + [
                        carry_table["region"]
                    ]
                carry_table = None

        # 检查页末表格是否跨页
        if regions and regions[-1].get("label") == TABLE_LABEL:
            last_html = _table_html(regions[-1])
            sig = _table_header_signature(last_html)
            # 启发式：末表且次页可能存在续表
            carry_table = {
                "region": regions[-1],
                "html": last_html,
                "_header_sig": sig,
            }
            regions = regions[:-1]

        page["regions"] = filter_dropped_regions(regions)
        merged_pages.append(page)

    if carry_table is not None and merged_pages:
        merged_pages[-1]["regions"] = list(merged_pages[-1].get("regions") or []) + [
            carry_table["region"]
        ]

    out = dict(document)
    out["pages"] = merged_pages
    out["postprocess"] = {**(document.get("postprocess") or {}), "cross_page_tables": True}
    return out


def _merge_table_html(prev_html: str, next_html: str) -> str:
    if not prev_html:
        return next_html
    if not next_html:
        return prev_html
    # 去掉 next 的 thead/首个 tr（重复表头）
    body_rows = re.findall(r"<tr[^>]*>.*?</tr>", next_html, re.I | re.S)
    if len(body_rows) <= 1:
        return prev_html
    next_body = "".join(body_rows[1:])
    prev_close = prev_html.rfind("</table>")
    if prev_close == -1:
        return prev_html + next_body
    return prev_html[:prev_close] + next_body + prev_html[prev_close:]


def apply_reading_order_to_page(page: Dict[str, Any]) -> Dict[str, Any]:
    """对单页 regions 应用 XY-Cut 阅读顺序（layout order 优先）。"""
    page = dict(page)
    regions = list(page.get("regions") or [])
    page_size = page.get("page_size") or []
    page_width = float(page_size[0]) if len(page_size) >= 1 else None
    boxes = [
        {
            "order": r.get("order"),
            "coordinate": r.get("coordinate"),
            "label": r.get("label"),
            "_region": r,
        }
        for r in regions
    ]
    sorted_boxes = sort_boxes_reading_order(boxes, page_width=page_width)
    page["regions"] = [b["_region"] for b in sorted_boxes]
    return page


def postprocess_document_ir(document: Dict[str, Any]) -> Dict[str, Any]:
    """IR 统一后处理入口。"""
    doc = dict(document)
    pages = []
    for page in doc.get("pages") or []:
        p = apply_reading_order_to_page(page)
        p["regions"] = filter_dropped_regions(p.get("regions") or [])
        pages.append(p)
    doc["pages"] = pages
    doc = merge_cross_page_tables(doc)
    return doc


def regions_to_plain_text(pages: List[Dict[str, Any]]) -> str:
    """从 IR 页列表拼接纯文本（已过滤 DROP）。"""
    parts: List[str] = []
    for page in pages:
        pn = int(page.get("page_index", 0)) + 1
        page_parts: List[str] = []
        for region in page.get("regions") or []:
            label = str(region.get("label") or "")
            if label in DROP_LABELS:
                continue
            content = region.get("content") or {}
            ctype = content.get("type")
            if ctype == "table":
                html = content.get("html") or region.get("text") or ""
                page_parts.append(html)
            elif ctype == "formula":
                latex = content.get("latex") or region.get("text") or ""
                if latex:
                    page_parts.append(f"$${latex}$$")
            elif ctype == "text":
                text = content.get("text") or region.get("text") or ""
                if text:
                    page_parts.append(text)
            elif region.get("text"):
                page_parts.append(str(region["text"]))
        if page_parts:
            parts.append(f"=== 第 {pn} 页 ===\n" + "\n".join(page_parts))
    return "\n\n".join(parts)
