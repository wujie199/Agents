"""PDF 转页图（pypdfium2）。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PDF_SUFFIX = ".pdf"


def _render_pdf_page(args: tuple[int, str, str, int]) -> str:
    import pypdfium2 as pdfium

    index, pdf_str, out_dir_str, dpi = args
    out_dir = Path(out_dir_str)
    doc = pdfium.PdfDocument(pdf_str)
    page = doc[index]
    bitmap = page.render(scale=dpi / 72.0)
    img_path = out_dir / f"page_{index:04d}.png"
    bitmap.to_pil().save(img_path)
    return str(img_path)


def pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    *,
    dpi: int = 200,
    pdf_threads: int = 2,
    max_pages: int | None = None,
) -> list[Path]:
    import pypdfium2 as pdfium

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    n = len(doc)
    if max_pages is not None:
        n = min(n, max_pages)

    if pdf_threads > 1 and n > 1:
        tasks = [(i, str(pdf_path), str(out_dir), dpi) for i in range(n)]
        paths_map: dict[int, Path] = {}
        with ThreadPoolExecutor(max_workers=min(pdf_threads, n)) as pool:
            futures = [pool.submit(_render_pdf_page, t) for t in tasks]
            for fut in as_completed(futures):
                p = Path(fut.result())
                idx = int(p.stem.split("_")[1])
                paths_map[idx] = p
        return [paths_map[i] for i in range(n)]

    paths: list[Path] = []
    scale = dpi / 72.0
    for i in range(n):
        bitmap = doc[i].render(scale=scale)
        img_path = out_dir / f"page_{i:04d}.png"
        bitmap.to_pil().save(img_path)
        paths.append(img_path)
    return paths


def resolve_image_paths(
    input_path: Path,
    pages_dir: Path,
    *,
    dpi: int = 200,
    pdf_threads: int = 2,
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
