"""
OCR 处理器 - 版面分析 + 文字识别 + 表格识别（PP-StructureV3 单 pipeline）

本地模型目录见 document/ocr/paths.py。
速度优化：单 pipeline 复用、无线表 e2e 结构识别、关闭非必要子模块、PDF 默认 150 DPI。
"""

import os
import sys
import re
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict

os.environ.setdefault("FLAGS_use_mkldnn", "0")

_OCR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _OCR_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from document.ocr.paths import (
    DEFAULT_TEST_PDF,
    LAYOUT,
    TEXT_REC,
    all_local_model_dirs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_LAYOUT_MODEL_DIR = str(LAYOUT)
DEFAULT_OCR_MODEL_DIR = str(TEXT_REC)
MACOS_MAX_IMAGE_SIDE = 1296


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
    """OCR处理结果"""
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
    """PDF文档OCR处理结果"""
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


class OCRProcessor:
    """OCR + 表格识别（单 PPStructureV3 pipeline，本地模型）"""

    def __init__(
        self,
        layout_model_dir: Optional[str] = DEFAULT_LAYOUT_MODEL_DIR,
        ocr_model_dir: Optional[str] = DEFAULT_OCR_MODEL_DIR,
        device: str = "cpu",
        use_gpu: bool = False,
        layout_model_name: str = "PP-DocLayoutV3",
        ocr_model_name: str = "PP-OCRv5_server_rec",
        use_table_recognition: bool = True,
        fast_mode: bool = True,
    ):
        self.layout_model_dir = layout_model_dir
        self.ocr_model_dir = ocr_model_dir
        self.device = device if not use_gpu else "gpu:0"
        self.layout_model_name = layout_model_name
        self.ocr_model_name = ocr_model_name
        self.use_table_recognition = use_table_recognition
        self.fast_mode = fast_mode

        self._pipeline = None
        self._predict_kwargs: Dict[str, Any] = {}

        logger.info(
            "初始化 OCR 处理器: device=%s, table=%s, fast=%s",
            self.device,
            use_table_recognition,
            fast_mode,
        )

    def _is_pdf(self, file_path: str) -> bool:
        return file_path.lower().endswith(".pdf")

    def _prepare_image(self, image_path: str) -> str:
        try:
            from PIL import Image
        except ImportError:
            return image_path

        with Image.open(image_path) as img:
            width, height = img.size
            max_side = max(width, height)
            if max_side <= MACOS_MAX_IMAGE_SIDE:
                return image_path

            ratio = MACOS_MAX_IMAGE_SIDE / max_side
            new_size = (int(width * ratio), int(height * ratio))
            logger.info(
                "图片 %dx%d 缩放至 %dx%d（macOS ARM 限制）",
                width,
                height,
                new_size[0],
                new_size[1],
            )
            resized = img.resize(new_size, Image.LANCZOS)
            temp_path = os.path.join(
                tempfile.gettempdir(),
                f"ocr_resized_{Path(image_path).stem}.png",
            )
            resized.save(temp_path, "PNG")
            return temp_path

    @staticmethod
    def _extract_result_dict(res: Any) -> Dict[str, Any]:
        if isinstance(res, dict):
            return res
        if hasattr(res, "res") and isinstance(res.res, dict):
            return res.res
        if hasattr(res, "keys"):
            return dict(res)
        return {}

    @staticmethod
    def _regions_from_ocr_dict(
        res_dict: Dict[str, Any],
        default_region_type: str = "text",
    ) -> List[LayoutRegion]:
        if "dt_polys" not in res_dict:
            return []

        dt_polys = res_dict["dt_polys"]
        rec_texts = res_dict.get("rec_texts", [])
        rec_scores = res_dict.get("rec_scores", [])

        regions = []
        for i, poly in enumerate(dt_polys):
            text = rec_texts[i] if i < len(rec_texts) else ""
            score = rec_scores[i] if i < len(rec_scores) else 0.9
            regions.append(
                LayoutRegion(
                    region_type=default_region_type,
                    bbox=poly.tolist() if hasattr(poly, "tolist") else list(poly),
                    confidence=float(score),
                    content=text,
                )
            )
        return regions

    def _build_model_kwargs(self) -> Dict[str, Any]:
        kwargs = all_local_model_dirs()
        if self.layout_model_dir:
            kwargs["layout_detection_model_dir"] = self.layout_model_dir
        if self.ocr_model_dir:
            kwargs["text_recognition_model_dir"] = self.ocr_model_dir
        return kwargs

    def _build_predict_kwargs(self) -> Dict[str, Any]:
        if self.fast_mode:
            return {
                "use_table_orientation_classify": False,
                "use_e2e_wireless_table_rec_model": True,
                "use_e2e_wired_table_rec_model": True,
                "text_det_limit_side_len": 960,
                "text_det_limit_type": "max",
            }
        return {
            "use_table_orientation_classify": False,
            "use_e2e_wireless_table_rec_model": True,
            "use_e2e_wired_table_rec_model": True,
            "text_det_limit_side_len": 1280,
            "text_det_limit_type": "max",
        }

    def _init_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        try:
            from paddleocr import PPStructureV3
        except ImportError as e:
            logger.error("请安装 paddleocr，并使用 conda py3.11 运行")
            raise ImportError("paddleocr 未安装") from e

        logger.info("加载 PP-StructureV3 pipeline（本地模型 + 表格识别）")
        model_kwargs = self._build_model_kwargs()
        self._predict_kwargs = self._build_predict_kwargs()

        self._pipeline = PPStructureV3(
            device=self.device,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_table_recognition=self.use_table_recognition,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_region_detection=False,
            **model_kwargs,
        )
        self._predict_kwargs["use_table_recognition"] = self.use_table_recognition
        logger.info("Pipeline 加载完成")

    @staticmethod
    def _block_to_dict(block: Any) -> Dict[str, Any]:
        if isinstance(block, dict):
            return block
        if hasattr(block, "keys"):
            return dict(block)
        return {
            "block_label": getattr(block, "block_label", None) or getattr(block, "label", "text"),
            "block_content": getattr(block, "block_content", None) or getattr(block, "content", ""),
        }

    def _parse_structure_page(
        self, res_dict: Dict[str, Any]
    ) -> Tuple[List[LayoutRegion], List[Dict[str, Any]], str]:
        regions: List[LayoutRegion] = []
        tables: List[Dict[str, Any]] = []
        text_parts: List[str] = []

        parsing = res_dict.get("parsing_res_list") or []
        if parsing:
            for block in parsing:
                b = self._block_to_dict(block)
                label = b.get("block_label") or b.get("label") or "text"
                content = b.get("block_content") or b.get("content") or ""
                if content:
                    text_parts.append(content)
                    regions.append(
                        LayoutRegion(
                            region_type=str(label),
                            bbox=[],
                            confidence=1.0,
                            content=content,
                        )
                    )
        else:
            ocr_res = res_dict.get("overall_ocr_res", res_dict)
            ocr_regions = self._regions_from_ocr_dict(ocr_res, "text")
            regions.extend(ocr_regions)
            text_parts.extend(r.content for r in ocr_regions if r.content)

        for idx, tbl in enumerate(res_dict.get("table_res_list") or []):
            if not isinstance(tbl, dict):
                tbl = dict(tbl) if hasattr(tbl, "keys") else {}
            html = tbl.get("pred_html") or ""
            plain = _html_to_text(html)
            tables.append({"index": idx, "html": html, "text": plain})
            regions.append(
                LayoutRegion(
                    region_type="table",
                    bbox=[],
                    confidence=1.0,
                    content=plain or html,
                    html=html,
                )
            )
            if plain or html:
                text_parts.append(plain or _html_to_text(html))

        full_text = "\n".join(p for p in text_parts if p)
        return regions, tables, full_text

    def _predict_page(self, image_path: str) -> Tuple[List[LayoutRegion], List[Dict[str, Any]], str]:
        self._init_pipeline()
        prepared = self._prepare_image(image_path)
        results = self._pipeline.predict(prepared, **self._predict_kwargs)

        all_regions: List[LayoutRegion] = []
        all_tables: List[Dict[str, Any]] = []
        texts: List[str] = []

        for res in results:
            res_dict = self._extract_result_dict(res)
            regions, tables, page_text = self._parse_structure_page(res_dict)
            all_regions.extend(regions)
            all_tables.extend(tables)
            if page_text:
                texts.append(page_text)

        return all_regions, all_tables, "\n\n".join(texts)

    def _pdf_to_images(self, pdf_path: str, dpi: int = 150, max_pages: Optional[int] = None) -> List[str]:
        try:
            from pdf2image import convert_from_path
        except ImportError:
            logger.error("请安装 pdf2image 与 poppler")
            raise ImportError("pdf2image 未安装") from None

        logger.info("PDF 转图片: %s, DPI=%s", pdf_path, dpi)
        kwargs: Dict[str, Any] = {"dpi": dpi}
        if max_pages is not None:
            kwargs["first_page"] = 1
            kwargs["last_page"] = max_pages
        images = convert_from_path(pdf_path, **kwargs)
        temp_dir = tempfile.mkdtemp(prefix="ocr_pdf_")
        paths = []
        for idx, image in enumerate(images):
            path = os.path.join(temp_dir, f"page_{idx + 1}.png")
            image.save(path, "PNG")
            paths.append(path)
        return paths

    def process_layout(self, image_path: str) -> List[LayoutRegion]:
        regions, _, _ = self._predict_page(image_path)
        return regions

    def process_ocr(self, image_path: str) -> List[LayoutRegion]:
        regions, _, _ = self._predict_page(image_path)
        return [r for r in regions if r.region_type != "table"]

    def process(
        self,
        input_path: str,
        use_layout: bool = True,
        use_ocr: bool = True,
        pdf_dpi: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> Union[OCRResult, PDFResult]:
        dpi = pdf_dpi if pdf_dpi is not None else (150 if self.fast_mode else 200)
        if self._is_pdf(input_path):
            return self.process_pdf(input_path, use_layout, use_ocr, dpi, max_pages)
        return self.process_image(input_path, use_layout, use_ocr)

    def process_image(
        self,
        image_path: str,
        use_layout: bool = True,
        use_ocr: bool = True,
    ) -> OCRResult:
        logger.info("处理图片: %s", image_path)
        if not use_layout and not use_ocr:
            return OCRResult(image_path=image_path, regions=[], full_text="")

        regions, tables, full_text = self._predict_page(image_path)

        metadata = {
            "use_layout": use_layout,
            "use_ocr": use_ocr,
            "use_table_recognition": self.use_table_recognition,
            "fast_mode": self.fast_mode,
            "pipeline": "PP-StructureV3",
            "region_count": len(regions),
            "table_count": len(tables),
        }

        return OCRResult(
            image_path=image_path,
            regions=regions,
            full_text=full_text,
            tables=tables,
            metadata=metadata,
        )

    def process_pdf(
        self,
        pdf_path: str,
        use_layout: bool = True,
        use_ocr: bool = True,
        dpi: int = 150,
        max_pages: Optional[int] = None,
    ) -> PDFResult:
        logger.info("处理 PDF: %s", pdf_path)
        image_paths = self._pdf_to_images(pdf_path, dpi, max_pages=max_pages)
        pages: List[OCRResult] = []
        all_texts: List[str] = []

        for idx, image_path in enumerate(image_paths):
            logger.info("处理第 %d/%d 页", idx + 1, len(image_paths))
            page = self.process_image(image_path, use_layout, use_ocr)
            page.page_number = idx + 1
            pages.append(page)
            all_texts.append(f"=== 第 {idx + 1} 页 ===\n{page.full_text}")

        return PDFResult(
            pdf_path=pdf_path,
            pages=pages,
            total_pages=len(pages),
            full_text="\n\n".join(all_texts),
            metadata={
                "use_layout": use_layout,
                "use_ocr": use_ocr,
                "dpi": dpi,
                "use_table_recognition": self.use_table_recognition,
                "fast_mode": self.fast_mode,
            },
        )

    def process_batch(
        self,
        image_paths: List[str],
        use_layout: bool = True,
        use_ocr: bool = True,
    ) -> List[OCRResult]:
        results = []
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
    layout_model_dir: Optional[str] = DEFAULT_LAYOUT_MODEL_DIR,
    ocr_model_dir: Optional[str] = DEFAULT_OCR_MODEL_DIR,
    use_gpu: bool = False,
    use_table_recognition: bool = True,
    fast_mode: bool = True,
) -> OCRProcessor:
    return OCRProcessor(
        layout_model_dir=layout_model_dir,
        ocr_model_dir=ocr_model_dir,
        use_gpu=use_gpu,
        use_table_recognition=use_table_recognition,
        fast_mode=fast_mode,
    )


if __name__ == "__main__":
    import argparse

    _PY311 = "/opt/miniconda3/envs/py3.11/bin/python"
    if sys.version_info[:2] != (3, 11):
        logger.error(
            "当前 Python: %s (%s)\n请使用: conda activate py3.11 && python document/ocr/processor.py\n或: %s document/ocr/processor.py",
            sys.executable,
            sys.version.split()[0],
            _PY311,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="OCR + 表格识别")
    parser.add_argument("input", nargs="?", default=DEFAULT_TEST_PDF)
    parser.add_argument("--layout-model", default=DEFAULT_LAYOUT_MODEL_DIR)
    parser.add_argument("--ocr-model", default=DEFAULT_OCR_MODEL_DIR)
    parser.add_argument("--output", "-o")
    parser.add_argument("--format", "-f", default="json", choices=["json", "txt"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--no-layout", action="store_true")
    parser.add_argument("--no-table", action="store_true", help="关闭表格识别（更快）")
    parser.add_argument("--quality", action="store_true", help="质量模式（较慢，DPI=200）")
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None, help="PDF 最多处理页数")
    args = parser.parse_args()

    input_file = args.input
    if not os.path.exists(input_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for path in (input_file, os.path.join(os.getcwd(), input_file), os.path.join(script_dir, input_file)):
            if os.path.exists(path):
                input_file = path
                break
        else:
            logger.error("输入文件不存在: %s", args.input)
            sys.exit(1)

    processor = create_processor(
        layout_model_dir=args.layout_model,
        ocr_model_dir=args.ocr_model,
        use_gpu=args.gpu,
        use_table_recognition=not args.no_table,
        fast_mode=not args.quality,
    )

    dpi = args.dpi
    if dpi is None:
        dpi = 200 if args.quality else 150

    result = processor.process(
        input_file,
        use_layout=not args.no_layout,
        use_ocr=True,
        pdf_dpi=dpi,
        max_pages=args.max_pages,
    )

    if hasattr(result, "pages"):
        logger.info("PDF 完成: %d 页, %d 字符", result.total_pages, len(result.full_text))
        for page in result.pages:
            tc = page.metadata.get("table_count", 0)
            logger.info("  第%d页: %d 区域, %d 表格", page.page_number, len(page.regions), tc)
    else:
        logger.info("图片完成: %d 区域, %d 表格", len(result.regions), len(result.tables))

    out = args.output or f"{Path(input_file).stem}_ocr.{args.format}"
    processor.save_result(result, out, args.format)

    print("\n" + "=" * 50)
    print("预览（前 500 字）:")
    print(result.full_text[:500])
    if len(result.full_text) > 500:
        print("...")
