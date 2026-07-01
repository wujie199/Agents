from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class IngestConfig:
    routing: str = "simplified"
    mode: str = "ocr_only"
    plain_text_formats: List[str] = field(default_factory=lambda: ["txt", "md"])
    ocr_backend: str = "auto"
    language: str = "ch"
    word_to_pdf: bool = True
    word_converter: str = "libreoffice"
    pdf_dpi: int = 200
    ocr_use_layout: bool = True
    enable_cleaning: bool = True
    ocr_postprocess: bool = True
    cleaning_level: str = "standard"
    ocr_model_root: Optional[str] = None
    ocr_device: str = "cpu"
    ocr_preprocess: str = "auto"
    ocr_enable_formula: bool = True
    ocr_formula_model: Optional[str] = None
    ocr_max_attempts: int = 3
    ocr_fast: bool = True
    ocr_table_e2e: bool = False
    ocr_enable_mkldnn: bool = True
    enable_header_footer_dedup: bool = False
    header_footer_threshold: float = 0.3
