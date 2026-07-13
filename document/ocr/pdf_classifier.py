# -*- coding: utf-8 -*-
"""PDF 页级/文档级路由信号与判别（原生 / 扫描 / 混合）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class PdfPageRoute(str, Enum):
    NATIVE = "native"
    SCAN = "scan"
    HYBRID = "hybrid"


class PdfDocRoute(str, Enum):
    NATIVE = "native"
    SCAN = "scan"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class PdfRouteConfig:
    """可配置阈值（与企业级「文字密度 / 可提取文字量」对齐）。"""

    min_native_chars: int = 80
    min_native_density: float = 0.0008  # chars / point^2
    max_scan_chars: int = 25
    max_scan_density: float = 0.0001
    hybrid_text_ratio: float = 0.35
    sample_pages: int = 0  # 0 = 全页


@dataclass
class PageSignals:
    page_index: int
    char_count: int
    page_width: float
    page_height: float
    text_density: float
    route: PdfPageRoute
    extractable: bool = True
    encrypted: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_area(self) -> float:
        return max(1.0, self.page_width * self.page_height)


@dataclass
class PdfClassification:
    doc_route: PdfDocRoute
    pages: List[PageSignals]
    total_pages: int
    native_pages: int
    scan_pages: int
    hybrid_pages: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_route": self.doc_route.value,
            "total_pages": self.total_pages,
            "native_pages": self.native_pages,
            "scan_pages": self.scan_pages,
            "hybrid_pages": self.hybrid_pages,
            "pages": [
                {
                    "page_index": p.page_index,
                    "char_count": p.char_count,
                    "text_density": round(p.text_density, 6),
                    "route": p.route.value,
                    "encrypted": p.encrypted,
                }
                for p in self.pages
            ],
        }


def _classify_page_signals(
    char_count: int,
    width: float,
    height: float,
    cfg: PdfRouteConfig,
    *,
    encrypted: bool = False,
) -> PdfPageRoute:
    if encrypted:
        return PdfPageRoute.SCAN
    area = max(1.0, width * height)
    density = char_count / area
    if char_count >= cfg.min_native_chars and density >= cfg.min_native_density:
        return PdfPageRoute.NATIVE
    if char_count <= cfg.max_scan_chars and density <= cfg.max_scan_density:
        return PdfPageRoute.SCAN
    return PdfPageRoute.HYBRID


def _aggregate_doc_route(pages: List[PageSignals]) -> PdfDocRoute:
    if not pages:
        return PdfDocRoute.SCAN
    routes = {p.route for p in pages}
    if routes == {PdfPageRoute.NATIVE}:
        return PdfDocRoute.NATIVE
    if PdfPageRoute.SCAN in routes and PdfPageRoute.NATIVE not in routes and PdfPageRoute.HYBRID not in routes:
        return PdfDocRoute.SCAN
    if PdfPageRoute.NATIVE in routes and PdfPageRoute.SCAN not in routes and PdfPageRoute.HYBRID not in routes:
        return PdfDocRoute.NATIVE
    return PdfDocRoute.HYBRID


def classify_pdf(pdf_path: Path, cfg: Optional[PdfRouteConfig] = None) -> PdfClassification:
    """分析 PDF 每页可提取文字量与密度，输出路由决策。"""
    cfg = cfg or PdfRouteConfig()
    path = Path(pdf_path)
    pages: List[PageSignals] = []

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("pypdfium2 未安装，无法进行 PDF 路由判别") from exc

    doc = pdfium.PdfDocument(str(path))
    n = len(doc)
    indices = range(n)
    if cfg.sample_pages and cfg.sample_pages < n:
        step = max(1, n // cfg.sample_pages)
        indices = list(range(0, n, step))[: cfg.sample_pages]

    for i in range(n):
        page = doc[i]
        width, height = page.get_size()
        encrypted = False
        char_count = 0
        try:
            textpage = page.get_textpage()
            char_count = textpage.count_chars()
            textpage.close()
        except Exception:
            encrypted = True
            char_count = 0

        if i not in indices and cfg.sample_pages:
            # 未采样页：沿用最近采样页路由（简化）
            ref = pages[-1] if pages else None
            route = ref.route if ref else PdfPageRoute.SCAN
            sig = PageSignals(
                page_index=i,
                char_count=char_count,
                page_width=float(width),
                page_height=float(height),
                text_density=char_count / max(1.0, width * height),
                route=route,
                encrypted=encrypted,
            )
        else:
            route = _classify_page_signals(
                char_count, float(width), float(height), cfg, encrypted=encrypted
            )
            sig = PageSignals(
                page_index=i,
                char_count=char_count,
                page_width=float(width),
                page_height=float(height),
                text_density=char_count / max(1.0, width * height),
                route=route,
                encrypted=encrypted,
            )
        pages.append(sig)

    doc.close()

    native = sum(1 for p in pages if p.route == PdfPageRoute.NATIVE)
    scan = sum(1 for p in pages if p.route == PdfPageRoute.SCAN)
    hybrid = sum(1 for p in pages if p.route == PdfPageRoute.HYBRID)

    return PdfClassification(
        doc_route=_aggregate_doc_route(pages),
        pages=pages,
        total_pages=len(pages),
        native_pages=native,
        scan_pages=scan,
        hybrid_pages=hybrid,
    )
