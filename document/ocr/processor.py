"""
OCR 处理器：委托 UniversalOcrPipeline（版面 → det/rec → 表格/公式 → QC）。

本地模型目录见 document/ocr/load_ocr.py（默认 OCR_MODEL_ROOT=/Volumes/wj/model/ocr）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

_OCR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _OCR_DIR.parents[1]


def _bootstrap_paddlex_env() -> None:
    """离线 Paddle 版面可视化：缓存目录 + 本地字体，避免下载 PingFang 失败。"""
    cache_home = _REPO_ROOT / "data" / "rag_offline" / ".paddlex"
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_home))
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(_REPO_ROOT / "data" / "rag_offline" / ".matplotlib"),
    )
    if os.environ.get("PADDLE_PDX_LOCAL_FONT_FILE_PATH"):
        return
    bundled = cache_home / "fonts" / "PingFang-SC-Regular.ttf"
    if bundled.is_file():
        os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = str(bundled)
        return
    for candidate in (
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
    ):
        if candidate.is_file():
            os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = str(candidate)
            return


_bootstrap_paddlex_env()

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from document.ocr.document_ir import document_to_markdown
from document.ocr.labels import DROP_LABELS
from document.ocr.load_ocr import (
    DEFAULT_TEST_PDF,
    FORMULA_MODEL_NAME,
    ensure_ocr_model_root,
    get_model_root,
)
from document.ocr.ocr_adapter import PipelineConfig
from document.ocr.pdf_utils import resolve_image_paths
from document.ocr.pipeline import UniversalOcrPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_LAYOUT_MODEL_DIR = str(get_model_root() / "PP-DocLayoutV3")
DEFAULT_OCR_MODEL_DIR = str(get_model_root() / "PP-OCRv5_server_rec")


@dataclass
class LayoutRegion:
    """版面/表格区域"""

    region_type: str
    bbox: List[List[int]]
    confidence: float
    content: str = ""
    html: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OCRResult:
    """OCR 处理结果"""

    image_path: str
    regions: List[LayoutRegion]
    full_text: str
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "regions": [r.to_dict() for r in self.regions],
            "full_text": self.full_text,
            "page_number": self.page_number,
            "metadata": self.metadata,
            "tables": self.tables,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class PDFResult:
    """PDF 文档 OCR 处理结果"""

    pdf_path: str
    pages: List[OCRResult]
    total_pages: int
    full_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "pages": [p.to_dict() for p in self.pages],
            "total_pages": self.total_pages,
            "full_text": self.full_text,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</t[dh]>", "\t", text, flags=re.I)
    text = re.sub(r"</tr>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _coord_to_bbox(coordinate: Any) -> List[List[int]]:
    if not coordinate or len(coordinate) < 4:
        return []
    xmin, ymin, xmax, ymax = (int(round(c)) for c in coordinate[:4])
    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ]


def _region_ir_to_layout(region: Dict[str, Any]) -> LayoutRegion:
    label = str(region.get("label") or "text")
    content = region.get("content") or {}
    ctype = content.get("type")
    text = ""
    html = ""
    tables: List[Dict[str, Any]] = []

    if ctype == "table" or label == "table":
        html = content.get("html") or region.get("text") or ""
        if region.get("table_result"):
            html = region["table_result"].get("pred_html") or html
        text = _html_to_text(html) or html
        if html:
            tables.append(
                {
                    "html": html,
                    "text": text,
                    "table_type": content.get("table_type")
                    or (region.get("table_result") or {}).get("table_type"),
                }
            )
    elif ctype == "formula":
        text = content.get("latex") or region.get("text") or ""
    elif ctype == "text":
        text = content.get("text") or region.get("text") or ""
    else:
        text = region.get("text") or ""

    score = region.get("rec_score")
    if score is None:
        score = region.get("layout_score")
    confidence = float(score) if score is not None else 1.0

    return LayoutRegion(
        region_type=label,
        bbox=_coord_to_bbox(region.get("coordinate")),
        confidence=confidence,
        content=text,
        html=html,
    )


def _page_ir_to_ocr_result(
    page: Dict[str, Any],
    *,
    fallback_image: str,
) -> OCRResult:
    regions: List[LayoutRegion] = []
    tables: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    for raw in page.get("regions") or []:
        if str(raw.get("label") or "") in DROP_LABELS:
            continue
        layout = _region_ir_to_layout(raw)
        regions.append(layout)
        if layout.region_type == "table" and layout.html:
            tables.append(
                {
                    "html": layout.html,
                    "text": layout.content or _html_to_text(layout.html),
                    "table_type": (raw.get("table_result") or {}).get("table_type"),
                }
            )
        if layout.content:
            text_parts.append(layout.content)

    image_path = page.get("processed_image") or page.get("image") or fallback_image
    qc = page.get("qc") or {}
    metadata = {
        "pipeline": "UniversalOcrPipeline",
        "region_count": len(regions),
        "table_count": len(tables),
        "qc_status": qc.get("status"),
        "attempt": page.get("attempt"),
        "preprocess": page.get("preprocess"),
    }

    return OCRResult(
        image_path=str(image_path),
        regions=regions,
        full_text="\n".join(text_parts),
        page_number=int(page.get("page_index", 0)) + 1,
        metadata=metadata,
        tables=tables,
    )


def _document_ir_to_pdf_result(
    document: Dict[str, Any],
    source_path: str,
) -> PDFResult:
    pages: List[OCRResult] = []
    all_texts: List[str] = []

    for page in document.get("pages") or []:
        ocr_page = _page_ir_to_ocr_result(page, fallback_image=source_path)
        pages.append(ocr_page)
        pn = ocr_page.page_number or len(pages)
        all_texts.append(f"=== 第 {pn} 页 ===\n{ocr_page.full_text}")

    qc_summary = document.get("qc_summary") or {}
    return PDFResult(
        pdf_path=source_path,
        pages=pages,
        total_pages=len(pages),
        full_text="\n\n".join(all_texts),
        metadata={
            "pipeline": "UniversalOcrPipeline",
            "pipeline_version": document.get("pipeline_version"),
            "qc_summary": qc_summary,
            "document_ir_schema": document.get("schema"),
            "document_ir": document,
            "pdf_classification": document.get("pdf_classification"),
        },
    )


class OCRProcessor:
    """OCR + 表格 + 公式（UniversalOcrPipeline，本地 Paddle 模型）。"""

    def __init__(
        self,
        layout_model_dir: Optional[str] = None,
        ocr_model_dir: Optional[str] = None,
        model_root: Optional[str] = None,
        device: str = "cpu",
        use_gpu: bool = False,
        layout_model_name: str = "PP-DocLayoutV3",
        ocr_model_name: str = "PP-OCRv5_server_rec",
        use_table_recognition: bool = True,
        fast_mode: bool = False,
        preprocess_mode: str = "auto",
        enable_formula: bool = True,
        formula_model_name: Optional[str] = None,
        max_attempts: int = 3,
        layout_threshold: float = 0.5,
        layout_score_threshold: float = 0.5,
        rec_batch_size: int = 8,
        crop_threads: int = 4,
        enable_mkldnn: bool = True,
        table_e2e: bool = False,
        pdf_threads: int = 1,
        enable_pdf_routing: bool = True,
    ):
        if model_root:
            root = get_model_root(model_root)
        elif layout_model_dir:
            root = Path(layout_model_dir).resolve().parent
        elif ocr_model_dir:
            root = Path(ocr_model_dir).resolve().parent
        else:
            root = get_model_root()

        ensure_ocr_model_root(root)
        self.model_root = root
        self.device = "gpu:0" if use_gpu else device
        self.use_table_recognition = use_table_recognition
        self.fast_mode = fast_mode
        self.preprocess_mode = preprocess_mode
        self.enable_formula = enable_formula
        self.pdf_threads = pdf_threads
        self.enable_pdf_routing = enable_pdf_routing
        self.layout_model_name = layout_model_name
        self.ocr_model_name = ocr_model_name

        self._cfg = PipelineConfig(
            model_root=root,
            device=self.device,
            threshold=layout_threshold,
            layout_score_threshold=layout_score_threshold,
            fast=fast_mode,
            rec_batch_size=max(1, rec_batch_size),
            crop_threads=max(1, crop_threads),
            enable_mkldnn=enable_mkldnn,
            table_e2e=table_e2e,
            preprocess_mode=preprocess_mode,
            enable_formula=enable_formula and use_table_recognition,
            formula_model_name=formula_model_name or FORMULA_MODEL_NAME,
            max_attempts=max(1, max_attempts),
        )
        self._pipeline = UniversalOcrPipeline(self._cfg)
        self._temp_dirs: List[str] = []

        logger.info(
            "初始化 OCR 处理器: root=%s device=%s table=%s fast=%s preprocess=%s",
            root,
            self.device,
            use_table_recognition,
            fast_mode,
            preprocess_mode,
        )

    def _is_pdf(self, file_path: str) -> bool:
        return file_path.lower().endswith(".pdf")

    def _run_pipeline(
        self,
        source: Path,
        *,
        dpi: int,
        max_pages: Optional[int] = None,
        enable_pdf_routing: bool = True,
    ) -> Dict[str, Any]:
        scratch_root = Path(tempfile.mkdtemp(prefix="ocr_scratch_"))
        self._temp_dirs.append(str(scratch_root))
        pages_dir = scratch_root / "pages"
        layout_out = None if self.fast_mode else scratch_root / "layout"
        crops_dir = None if self.fast_mode else scratch_root / "crops"
        work_root = scratch_root / "work"

        if source.suffix.lower() == ".pdf" and enable_pdf_routing:
            from document.ocr.pdf_router import process_pdf_with_routing

            # #region agent log
            try:
                from document.rag.shared.debug_trace import debug_trace

                debug_trace(
                    "ocr/processor.py:_run_pipeline",
                    "PDF 混合路由开始",
                    data={"source": str(source), "dpi": dpi, "max_pages": max_pages},
                    hypothesis_id="H2",
                )
            except ImportError:
                pass
            # #endregion

            def _ocr_page(**kwargs: Any) -> Dict[str, Any]:
                scratch = kwargs.get("scratch_dir")
                if scratch is not None:
                    scratch = Path(scratch)
                    scratch.mkdir(parents=True, exist_ok=True)
                return self._pipeline.process_single_page(
                    image_path=kwargs["image_path"],
                    page_index=kwargs["page_index"],
                    layout_out=kwargs.get("layout_out"),
                    crops_dir=kwargs.get("crops_dir"),
                    scratch_dir=scratch or (work_root / f"page_{kwargs['page_index']:04d}"),
                )

            document, _classification = process_pdf_with_routing(
                source,
                ocr_page_fn=_ocr_page,
                dpi=dpi,
                pdf_threads=self.pdf_threads,
                max_pages=max_pages,
                layout_out=layout_out,
                crops_dir=crops_dir,
                scratch_root=work_root,
            )
            document["_plain_markdown"] = document_to_markdown(
                document, title=source.name
            )
            # #region agent log
            try:
                from document.rag.shared.debug_trace import debug_trace, preview_text

                pages = document.get("pages") or []
                debug_trace(
                    "ocr/processor.py:_run_pipeline:pdf_routed",
                    "PDF 路由 + OCR 完成",
                    data={
                        "source": str(source),
                        "pages": len(pages),
                        "regions_total": sum(len(p.get("regions") or []) for p in pages),
                        "classification": document.get("classification"),
                        "markdown_chars": len(document.get("_plain_markdown") or ""),
                        "markdown_preview": preview_text(
                            document.get("_plain_markdown") or "", 500
                        ),
                    },
                    hypothesis_id="H2",
                )
            except ImportError:
                pass
            # #endregion
            return document

        image_paths = resolve_image_paths(
            source,
            pages_dir,
            dpi=dpi,
            pdf_threads=self.pdf_threads,
            max_pages=max_pages,
        )

        document = self._pipeline.process_all_pages(
            image_paths,
            source=source,
            layout_out=layout_out,
            crops_dir=crops_dir,
            scratch_root=work_root,
        )
        from document.ocr.ir_postprocess import postprocess_document_ir

        document = postprocess_document_ir(document)
        document["_plain_markdown"] = document_to_markdown(
            document, title=source.name
        )
        # #region agent log
        try:
            from document.rag.shared.debug_trace import debug_trace, preview_text

            pages = document.get("pages") or []
            debug_trace(
                "ocr/processor.py:_run_pipeline:image_path",
                "逐页 OCR + IR 后处理完成",
                data={
                    "source": str(source),
                    "pages": len(pages),
                    "regions_total": sum(len(p.get("regions") or []) for p in pages),
                    "postprocess": document.get("postprocess"),
                    "markdown_chars": len(document.get("_plain_markdown") or ""),
                    "markdown_preview": preview_text(
                        document.get("_plain_markdown") or "", 500
                    ),
                },
                hypothesis_id="H2",
            )
        except ImportError:
            pass
        # #endregion
        return document

    def process(
        self,
        input_path: str,
        use_layout: bool = True,
        use_ocr: bool = True,
        pdf_dpi: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> Union[OCRResult, PDFResult]:
        if not use_layout and not use_ocr:
            if self._is_pdf(input_path):
                return PDFResult(
                    pdf_path=input_path,
                    pages=[],
                    total_pages=0,
                    full_text="",
                )
            return OCRResult(image_path=input_path, regions=[], full_text="")

        dpi = pdf_dpi if pdf_dpi is not None else (150 if self.fast_mode else 200)
        source = Path(input_path)
        document = self._run_pipeline(source, dpi=dpi, max_pages=max_pages, enable_pdf_routing=self.enable_pdf_routing)

        if self._is_pdf(input_path):
            result = _document_ir_to_pdf_result(document, input_path)
            result.metadata["document_ir"] = document
            result.metadata["dpi"] = dpi
            result.metadata["use_layout"] = use_layout
            result.metadata["use_ocr"] = use_ocr
            result.metadata["use_table_recognition"] = self.use_table_recognition
            result.metadata["fast_mode"] = self.fast_mode
            return result

        pages = document.get("pages") or []
        if not pages:
            return OCRResult(image_path=input_path, regions=[], full_text="")

        page = pages[0]
        ocr_result = _page_ir_to_ocr_result(page, fallback_image=input_path)
        ocr_result.metadata["document_ir"] = document
        ocr_result.metadata["use_layout"] = use_layout
        ocr_result.metadata["use_ocr"] = use_ocr
        ocr_result.metadata["use_table_recognition"] = self.use_table_recognition
        ocr_result.metadata["fast_mode"] = self.fast_mode
        return ocr_result

    def process_layout(self, image_path: str) -> List[LayoutRegion]:
        result = self.process_image(image_path, use_layout=True, use_ocr=False)
        return result.regions

    def process_ocr(self, image_path: str) -> List[LayoutRegion]:
        result = self.process_image(image_path, use_layout=True, use_ocr=True)
        return [r for r in result.regions if r.region_type != "table"]

    def process_image(
        self,
        image_path: str,
        use_layout: bool = True,
        use_ocr: bool = True,
    ) -> OCRResult:
        result = self.process(
            image_path,
            use_layout=use_layout,
            use_ocr=use_ocr,
        )
        if isinstance(result, PDFResult):
            return result.pages[0] if result.pages else OCRResult(
                image_path=image_path, regions=[], full_text=""
            )
        return result

    def process_pdf(
        self,
        pdf_path: str,
        use_layout: bool = True,
        use_ocr: bool = True,
        dpi: int = 150,
        max_pages: Optional[int] = None,
    ) -> PDFResult:
        result = self.process(
            pdf_path,
            use_layout=use_layout,
            use_ocr=use_ocr,
            pdf_dpi=dpi,
            max_pages=max_pages,
        )
        assert isinstance(result, PDFResult)
        return result

    def process_batch(
        self,
        image_paths: List[str],
        use_layout: bool = True,
        use_ocr: bool = True,
    ) -> List[OCRResult]:
        results: List[OCRResult] = []
        for idx, path in enumerate(image_paths):
            logger.info("批量进度 %d/%d", idx + 1, len(image_paths))
            try:
                results.append(self.process_image(path, use_layout, use_ocr))
            except Exception as exc:
                logger.error("处理失败 %s: %s", path, exc)
                results.append(
                    OCRResult(
                        image_path=path,
                        regions=[],
                        full_text="",
                        metadata={"error": str(exc)},
                    )
                )
        return results

    def save_result(
        self,
        result: Union[OCRResult, PDFResult],
        output_path: str,
        format: str = "json",
    ) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            output_file.write_text(result.to_json(), encoding="utf-8")
        elif format == "txt":
            output_file.write_text(result.full_text, encoding="utf-8")
        else:
            raise ValueError(f"不支持的格式: {format}")
        logger.info("结果已保存: %s", output_file)


def create_processor(
    layout_model_dir: Optional[str] = None,
    ocr_model_dir: Optional[str] = None,
    model_root: Optional[str] = None,
    use_gpu: bool = False,
    use_table_recognition: bool = True,
    fast_mode: bool = False,
    **kwargs: Any,
) -> OCRProcessor:
    return OCRProcessor(
        layout_model_dir=layout_model_dir,
        ocr_model_dir=ocr_model_dir,
        model_root=model_root,
        use_gpu=use_gpu,
        use_table_recognition=use_table_recognition,
        fast_mode=fast_mode,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR + 表格/公式识别（UniversalOcrPipeline）")
    parser.add_argument("input", nargs="?", default=DEFAULT_TEST_PDF)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--output", "-o")
    parser.add_argument("--format", "-f", default="json", choices=["json", "txt"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-layout", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument(
        "--preprocess",
        choices=("auto", "off", "on"),
        default="auto",
    )
    parser.add_argument("--no-formula", action="store_true")
    args = parser.parse_args()

    input_file = args.input
    if not os.path.exists(input_file):
        for path in (
            input_file,
            os.path.join(os.getcwd(), input_file),
            os.path.join(str(_OCR_DIR), input_file),
        ):
            if os.path.exists(path):
                input_file = path
                break
        else:
            logger.error("输入文件不存在: %s", args.input)
            sys.exit(1)

    root = str(args.model_root) if args.model_root else None
    dpi = args.dpi or (200 if args.quality else 150)

    processor = create_processor(
        model_root=root,
        use_gpu=args.gpu,
        device=args.device,
        use_table_recognition=not args.no_table,
        fast_mode=not args.quality,
        preprocess_mode=args.preprocess,
        enable_formula=not args.no_formula,
    )

    result = processor.process(
        input_file,
        use_layout=not args.no_layout,
        use_ocr=True,
        pdf_dpi=dpi,
        max_pages=args.max_pages,
    )

    if hasattr(result, "pages"):
        logger.info("PDF 完成: %d 页, %d 字符", result.total_pages, len(result.full_text))
    else:
        logger.info(
            "图片完成: %d 区域, %d 表格",
            len(result.regions),
            len(result.tables),
        )

    out = args.output or f"{Path(input_file).stem}_ocr.{args.format}"
    processor.save_result(result, out, args.format)

    print("\n" + "=" * 50)
    print("预览（前 500 字）:")
    print(result.full_text[:500])
    if len(result.full_text) > 500:
        print("...")
