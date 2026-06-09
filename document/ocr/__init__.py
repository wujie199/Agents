"""文档 OCR：版面分析、文字、表格与公式识别（UniversalOcrPipeline）。"""

from document.ocr.load_ocr import get_model_root, validate_core_models
from document.ocr.processor import OCRProcessor, create_processor
from document.ocr.pipeline import UniversalOcrPipeline

__all__ = [
    "OCRProcessor",
    "UniversalOcrPipeline",
    "create_processor",
    "get_model_root",
    "validate_core_models",
]
