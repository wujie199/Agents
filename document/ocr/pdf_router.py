# -*- coding: utf-8 -*-
"""PDF 混合路由：原生抽字 + 扫描 OCR + 合并 document_ir。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from document.ocr.document_ir import build_document_ir, normalize_page
from document.ocr.ir_postprocess import postprocess_document_ir, regions_to_plain_text
from document.ocr.pdf_classifier import (
    PdfClassification,
    PdfPageRoute,
    PdfRouteConfig,
    classify_pdf,
)
from document.ocr.pdf_native import extract_native_page, extract_native_pages
from document.ocr.pdf_utils import resolve_image_paths
from document.rag.shared.debug_trace import (
    preview_text,
    summarize_ocr_regions,
    trace_pipeline_step,
)


def process_pdf_with_routing(
    pdf_path: Path,
    *,
    ocr_page_fn: Callable[..., Dict[str, Any]],
    dpi: int = 200,
    pdf_threads: int = 1,
    max_pages: Optional[int] = None,
    route_cfg: Optional[PdfRouteConfig] = None,
    layout_out: Optional[Path] = None,
    crops_dir: Optional[Path] = None,
    scratch_root: Optional[Path] = None,
) -> tuple[Dict[str, Any], PdfClassification]:
    """
    P0 入口：按页路由 native / scan / hybrid。
    hybrid 页当前降级为全页 OCR（后续可接局部 OCR）。
    """
    classification = classify_pdf(pdf_path, route_cfg)
    # #region agent log
    trace_pipeline_step(
        "ocr",
        "pdf_classify",
        "PDF 页路由分类",
        data={
            "pdf": str(pdf_path),
            "total_pages": len(classification.pages),
            "native_pages": sum(1 for p in classification.pages if p.route == PdfPageRoute.NATIVE),
            "scan_pages": sum(1 for p in classification.pages if p.route == PdfPageRoute.SCAN),
            "hybrid_pages": sum(1 for p in classification.pages if p.route == PdfPageRoute.HYBRID),
        },
        artifact={
            "classification": classification.to_dict(),
            "page_routes": [
                {"page_index": p.page_index, "route": p.route.value, "reason": p.reason}
                for p in classification.pages
            ],
        },
        hypothesis_id="H2",
    )
    # #endregion

    pages_dir = (scratch_root / "pages") if scratch_root else pdf_path.parent / "_pages"
    image_paths = resolve_image_paths(
        pdf_path,
        pages_dir,
        dpi=dpi,
        pdf_threads=pdf_threads,
        max_pages=max_pages,
    )
    # #region agent log
    trace_pipeline_step(
        "ocr",
        "pdf_render",
        "PDF 转页图完成",
        data={
            "pdf": str(pdf_path),
            "dpi": dpi,
            "page_images": len(image_paths),
        },
        artifact={"image_paths": [str(p) for p in image_paths]},
        hypothesis_id="H2",
    )
    # #endregion

    native_indices = [
        p.page_index
        for p in classification.pages
        if p.route == PdfPageRoute.NATIVE
    ]
    native_by_index = extract_native_pages(pdf_path, native_indices)

    raw_pages: List[Dict[str, Any]] = []
    for page_index, image_path in enumerate(image_paths):
        sig = classification.pages[page_index] if page_index < len(classification.pages) else None
        route = sig.route if sig else PdfPageRoute.SCAN

        if route == PdfPageRoute.NATIVE:
            page = native_by_index.get(page_index) or extract_native_page(pdf_path, page_index)
            page = normalize_page(page)
            page["image"] = str(image_path.resolve())
            page["route"] = "native"
        else:
            page = ocr_page_fn(
                image_path=image_path,
                page_index=page_index,
                layout_out=layout_out,
                crops_dir=crops_dir,
                scratch_dir=(scratch_root / f"page_{page_index:04d}") if scratch_root else None,
            )
            page = normalize_page(page)
            page["route"] = "hybrid" if route == PdfPageRoute.HYBRID else "scan"
            page.setdefault("image", str(image_path.resolve()))

        raw_pages.append(page)
        # #region agent log
        trace_pipeline_step(
            "ocr",
            f"page_{page_index:04d}_{route.value if sig else 'scan'}",
            f"PDF 第 {page_index + 1} 页处理完成",
            data={
                "page_index": page_index,
                "route": page.get("route"),
                "regions": len(page.get("regions") or []),
                "qc_status": (page.get("qc") or {}).get("status"),
            },
            artifact={
                "page_index": page_index,
                "route": page.get("route"),
                "image": page.get("image"),
                "qc": page.get("qc"),
                "regions": summarize_ocr_regions(page.get("regions") or []),
            },
            hypothesis_id="H2",
        )
        # #endregion

    document = build_document_ir(
        source=pdf_path,
        pages=raw_pages,
        pipeline_version="1.1.0-routed",
        config_summary={
            "pdf_routing": classification.to_dict(),
            "dpi": dpi,
        },
    )
    document["pdf_classification"] = classification.to_dict()
    document = postprocess_document_ir(document)
    document["_plain_text"] = regions_to_plain_text(document.get("pages") or [])
    # #region agent log
    trace_pipeline_step(
        "ocr",
        "pdf_document_ir",
        "PDF document_ir 合并完成",
        data={
            "pdf": str(pdf_path),
            "pages": len(document.get("pages") or []),
            "plain_text_chars": len(document.get("_plain_text") or ""),
            "plain_text_preview": preview_text(document.get("_plain_text") or "", 500),
        },
        artifact={
            "pipeline_version": document.get("pipeline_version"),
            "config_summary": document.get("config_summary"),
            "pdf_classification": document.get("pdf_classification"),
            "page_summaries": [
                {
                    "page_index": p.get("page_index"),
                    "route": p.get("route"),
                    "region_count": len(p.get("regions") or []),
                    "qc_status": (p.get("qc") or {}).get("status"),
                }
                for p in (document.get("pages") or [])
            ],
        },
        hypothesis_id="H2",
    )
    # #endregion
    return document, classification
