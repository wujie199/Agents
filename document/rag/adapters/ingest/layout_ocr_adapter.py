import io
import logging
from typing import Optional, Dict, Any, BinaryIO, List, Tuple

from core.ports.ingest import (
    IngestPort,
    IngestResult,
    IngestConfig,
    IngestStatus,
    DocumentFormat,
)


class LayoutOCRAdapter:
    """Layout analysis + OCR for PDF, images, and HTML."""
    
    def __init__(
        self,
        ocr_backend: str = "auto",
        use_layout_analysis: bool = True,
        use_gpu: bool = False,
        language: str = "ch"
    ):
        self._logger = logging.getLogger("ingest.layout_ocr")
        self._use_layout = use_layout_analysis
        self._use_gpu = use_gpu
        self._language = language
        self._ocr_backend = self._detect_ocr_backend(ocr_backend)
        self._ocr_engine = None
        self._layout_engine = None
    
    def _detect_ocr_backend(self, preference: str) -> str:
        if preference != "auto":
            return preference
        
        try:
            from paddleocr import PaddleOCR
            return "paddleocr"
        except ImportError:
            pass
        
        try:
            import pytesseract
            return "tesseract"
        except ImportError:
            pass
        
        try:
            import easyocr
            return "easyocr"
        except ImportError:
            pass
        
        self._logger.warning("No OCR backend available")
        return "none"
    
    def _init_ocr_engine(self) -> Any:
        if self._ocr_engine is not None:
            return self._ocr_engine
        
        if self._ocr_backend == "paddleocr":
            from paddleocr import PaddleOCR
            
            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="ch" if self._language in ("ch", "zh") else "en",
                use_gpu=self._use_gpu,
                show_log=False,
                structure_version="PP-StructureV2" if self._use_layout else None,
            )
        
        elif self._ocr_backend == "tesseract":
            import pytesseract
            self._ocr_engine = pytesseract
        
        elif self._ocr_backend == "easyocr":
            import easyocr
            
            langs = ["ch_sim", "en"] if self._language in ("ch", "zh") else ["en"]
            self._ocr_engine = easyocr.Reader(langs, gpu=self._use_gpu)
        
        return self._ocr_engine
    
    def ingest(
        self,
        source: BinaryIO,
        doc_format: DocumentFormat,
        doc_id: str,
        config: Optional[IngestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestResult:
        config = config or IngestConfig()
        metadata = metadata or {}
        
        metadata["doc_id"] = doc_id
        metadata["doc_format"] = doc_format.value
        metadata["ocr_backend"] = self._ocr_backend
        metadata["use_layout"] = self._use_layout
        
        if self._ocr_backend == "none":
            return IngestResult(
                content="",
                metadata=metadata,
                status=IngestStatus.FAILED,
                errors=["No OCR backend available"]
            )
        
        try:
            images = self._convert_to_images(source, doc_format, config)
            
            if not images:
                return IngestResult(
                    content="",
                    metadata=metadata,
                    status=IngestStatus.FAILED,
                    errors=["Failed to convert document to images"]
                )
            
            return self._process_images(images, config, metadata)
            
        except Exception as e:
            self._logger.error(f"Layout OCR failed: {e}")
            return IngestResult(
                content="",
                metadata=metadata,
                status=IngestStatus.FAILED,
                errors=[str(e)]
            )
    
    def ingest_from_path(
        self,
        file_path: str,
        doc_id: str,
        config: Optional[IngestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestResult:
        metadata = metadata or {}
        metadata["source_path"] = file_path
        
        with open(file_path, "rb") as f:
            doc_format = self._detect_format(file_path)
            return self.ingest(f, doc_format, doc_id, config, metadata)
    
    def _detect_format(self, file_path: str) -> DocumentFormat:
        import os
        ext = os.path.splitext(file_path)[1].lower()
        
        format_map = {
            ".pdf": DocumentFormat.PDF,
            ".png": DocumentFormat.IMAGE,
            ".jpg": DocumentFormat.IMAGE,
            ".jpeg": DocumentFormat.IMAGE,
            ".bmp": DocumentFormat.IMAGE,
            ".tiff": DocumentFormat.IMAGE,
            ".tif": DocumentFormat.IMAGE,
            ".html": DocumentFormat.HTML,
            ".htm": DocumentFormat.HTML,
        }
        
        return format_map.get(ext, DocumentFormat.PDF)
    
    def supports_format(self, doc_format: DocumentFormat) -> bool:
        return doc_format in [
            DocumentFormat.PDF,
            DocumentFormat.IMAGE,
            DocumentFormat.HTML,
        ]
    
    def _convert_to_images(
        self,
        source: BinaryIO,
        doc_format: DocumentFormat,
        config: IngestConfig
    ) -> List[Tuple[Any, int]]:
        """Module docstring."""
        from PIL import Image
        
        images = []
        
        if doc_format == DocumentFormat.IMAGE:
            source.seek(0)
            img = Image.open(source)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            images.append((img, 1))
        
        elif doc_format == DocumentFormat.PDF:
            images = self._pdf_to_images(source, config)
        
        elif doc_format == DocumentFormat.HTML:
            images = self._html_to_images(source, config)
        
        return images
    
    def _pdf_to_images(
        self,
        source: BinaryIO,
        config: IngestConfig
    ) -> List[Tuple[Any, int]]:
        """Module helper."""
        try:
            import fitz
        except ImportError:
            self._logger.error("PyMuPDF not available for PDF conversion")
            return []
        
        source.seek(0)
        doc = fitz.open(stream=source.read(), filetype="pdf")
        
        images = []
        dpi = config.dpi
        max_pages = config.max_pages or len(doc)
        
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            
            img_data = pix.tobytes("png")
            from PIL import Image
            img = Image.open(io.BytesIO(img_data))
            
            images.append((img, page_num + 1))
        
        doc.close()
        return images
    
    def _html_to_images(
        self,
        source: BinaryIO,
        config: IngestConfig
    ) -> List[Tuple[Any, int]]:
        """Render HTML to images."""
        try:
            from imgkit import from_string
        except ImportError:
            self._logger.warning("imgkit not available, using text extraction")
            return self._html_text_fallback(source)
        
        source.seek(0)
        html_content = source.read().decode('utf-8', errors='ignore')
        
        try:
            img_data = from_string(html_content, False)
            from PIL import Image
            img = Image.open(io.BytesIO(img_data))
            return [(img, 1)]
        except Exception as e:
            self._logger.warning(f"HTML rendering failed: {e}")
            return self._html_text_fallback(source)
    
    def _html_text_fallback(
        self,
        source: BinaryIO
    ) -> List[Tuple[Any, int]]:
        """HTML text fallback when rendering fails."""
        source.seek(0)
        html_content = source.read().decode('utf-8', errors='ignore')
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            
            text = soup.get_text(separator="\n")
            
            from PIL import Image
            img = Image.new('RGB', (800, 100), color='white')
            return [(img, 1)]
            
        except ImportError:
            return []
    
    def _process_images(
        self,
        images: List[Tuple[Any, int]],
        config: IngestConfig,
        metadata: Dict[str, Any]
    ) -> IngestResult:
        """Module docstring."""
        import numpy as np
        
        pages_content = []
        all_tables = []
        all_images = []
        
        engine = self._init_ocr_engine()
        
        for img, page_num in images:
            if self._ocr_backend == "paddleocr" and self._use_layout:
                page_result = self._process_with_layout(img, engine, page_num)
            else:
                page_result = self._process_with_ocr(img, engine, page_num, config)
            
            pages_content.append(page_result["page"])
            all_tables.extend(page_result.get("tables", []))
            all_images.extend(page_result.get("images", []))
        
        full_content = "\n\n".join(p["content"] for p in pages_content)
        
        metadata["page_count"] = len(pages_content)
        metadata["table_count"] = len(all_tables)
        metadata["image_count"] = len(all_images)
        
        return IngestResult(
            content=full_content,
            metadata=metadata,
            status=IngestStatus.SUCCESS,
            pages=pages_content,
            tables=all_tables,
            images=all_images,
        )
    
    def _process_with_layout(
        self,
        img: Any,
        engine: Any,
        page_num: int
    ) -> Dict[str, Any]:
        """Module docstring."""
        import numpy as np
        
        img_array = np.array(img)
        
        result = engine.ocr(img_array, cls=True)
        
        lines = []
        boxes = []
        
        if result and result[0]:
            for item in result[0]:
                box = item[0]
                text_info = item[1]
                text = text_info[0]
                confidence = text_info[1]
                
                lines.append(text)
                boxes.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [int(x) for point in box for x in point],
                })
        
        content = "\n".join(lines)
        
        return {
            "page": {
                "page_num": page_num,
                "content": content,
                "char_count": len(content),
                "boxes": boxes,
            },
            "tables": [],
            "images": [],
        }
    
    def _process_with_ocr(
        self,
        img: Any,
        engine: Any,
        page_num: int,
        config: IngestConfig
    ) -> Dict[str, Any]:
        """Module helper."""
        lines = []
        
        if self._ocr_backend == "tesseract":
            lang = "chi_sim+eng" if config.language in ("ch", "zh") else "eng"
            
            data = engine.image_to_data(
                img,
                lang=lang,
                output_type=engine.Output.DICT
            )
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = data['conf'][i]
                
                if text and conf > 30:
                    lines.append(text)
        
        elif self._ocr_backend == "easyocr":
            import numpy as np
            img_array = np.array(img)
            
            results = engine.readtext(img_array)
            lines = [item[1] for item in results if item[2] > 0.5]
        
        content = "\n".join(lines)
        
        return {
            "page": {
                "page_num": page_num,
                "content": content,
                "char_count": len(content),
            },
            "tables": [],
            "images": [],
        }
