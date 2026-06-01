from typing import Protocol, List, Dict, Any, Optional, BinaryIO
from dataclasses import dataclass, field
from enum import Enum


class IngestStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class DocumentFormat(str, Enum):
    PDF = "pdf"
    WORD = "word"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    IMAGE = "image"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"


@dataclass
class IngestResult:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: IngestStatus = IngestStatus.SUCCESS
    pages: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def page_count(self) -> int:
        return len(self.pages)
    
    @property
    def has_tables(self) -> bool:
        return len(self.tables) > 0
    
    @property
    def has_images(self) -> bool:
        return len(self.images) > 0


@dataclass
class IngestConfig:
    extract_tables: bool = True
    extract_images: bool = False
    ocr_fallback: bool = True
    preserve_formatting: bool = True
    language: str = "zh"
    dpi: int = 200
    max_pages: Optional[int] = None


class IngestPort(Protocol):
    def ingest(
        self,
        source: BinaryIO,
        doc_format: DocumentFormat,
        doc_id: str,
        config: Optional[IngestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestResult:
        ...
    
    def ingest_from_path(
        self,
        file_path: str,
        doc_id: str,
        config: Optional[IngestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestResult:
        ...
    
    def supports_format(self, doc_format: DocumentFormat) -> bool:
        ...
