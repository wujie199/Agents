"""PDF 转页图（pypdfium2）。"""

from __future__ import annotations

import logging
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PDF_SUFFIX = ".pdf"

_log = logging.getLogger(__name__)


def default_pdf_threads() -> int:
    """pypdfium2 并发打开同一 PDF 不稳定，默认单线程渲染。"""
    return 1


def _render_page_to_path(doc: object, index: int, out_dir: Path, dpi: int) -> Path:
    scale = dpi / 72.0
    page = doc[index]
    bitmap = page.render(scale=scale)
    img_path = out_dir / f"page_{index:04d}.png"
    try:
        bitmap.to_pil().save(img_path)
    finally:
        close = getattr(bitmap, "close", None)
        if callable(close):
            close()
    return img_path


def pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    *,
    dpi: int = 200,
    pdf_threads: int = 1,
    max_pages: int | None = None,
) -> list[Path]:
    import pypdfium2 as pdfium

    if pdf_threads > 1:
        _log.warning(
            "pdf_threads=%d 已忽略：pypdfium2 不支持安全的多线程渲染，回退单线程",
            pdf_threads,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        n = len(doc)
        if max_pages is not None:
            n = min(n, max_pages)
        paths = [_render_page_to_path(doc, i, out_dir, dpi) for i in range(n)]
    finally:
        doc.close()
    # #region agent log
    try:
        from document.rag.shared.debug_trace import trace_pipeline_step

        trace_pipeline_step(
            "ocr",
            "pdf_to_images",
            "pypdfium2 渲染页图",
            data={"pdf": str(pdf_path), "dpi": dpi, "pages": len(paths)},
            artifact={"output_paths": [str(p) for p in paths]},
            hypothesis_id="H2",
        )
    except ImportError:
        pass
    # #endregion
    return paths


def resolve_image_paths(
    input_path: Path,
    pages_dir: Path,
    *,
    dpi: int = 200,
    pdf_threads: int = 1,
    max_pages: int | None = None,
) -> list[Path]:
    suffix = input_path.suffix.lower()
    if suffix == PDF_SUFFIX:
        return pdf_to_images(
            input_path,
            pages_dir,
            dpi=dpi,
            pdf_threads=pdf_threads,
            max_pages=max_pages,
        )
    if suffix in IMAGE_SUFFIXES:
        return [input_path]
    raise ValueError(f"不支持的输入格式: {suffix}")
