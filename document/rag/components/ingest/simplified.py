import io
import logging
import os
from typing import Optional, Dict, Any, BinaryIO

from core.ports.ingest import (
    IngestPort,
    IngestResult,
    IngestConfig,
    IngestStatus,
    DocumentFormat,
)
from document.rag.components.ingest.word import WordIngestAdapter
from document.rag.components.ingest.layout_ocr import LayoutOCRAdapter


class SimplifiedIngestAdapter:
    """Simplified ingest: Word or layout+OCR."""
    
    def __init__(
        self,
        word_adapter: Optional[WordIngestAdapter] = None,
        layout_ocr_adapter: Optional[LayoutOCRAdapter] = None,
        use_gpu: bool = False,
        ocr_backend: str = "auto",
        language: str = "ch"
    ):
        self._logger = logging.getLogger("ingest.simplified")
        
        self._word_adapter = word_adapter or WordIngestAdapter()
        self._layout_ocr_adapter = layout_ocr_adapter or LayoutOCRAdapter(
            ocr_backend=ocr_backend,
            use_gpu=use_gpu,
            language=language
        )
    
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
        
        if doc_format == DocumentFormat.WORD:
            self._logger.info(f"Processing {doc_id} with Word adapter")
            return self._word_adapter.ingest(
                source, doc_format, doc_id, config, metadata
            )
        else:
            self._logger.info(f"Processing {doc_id} with Layout OCR adapter")
            return self._layout_ocr_adapter.ingest(
                source, doc_format, doc_id, config, metadata
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
        
        doc_format = self._detect_format(file_path)
        
        if doc_format is None:
            return IngestResult(
                content="",
                metadata=metadata,
                status=IngestStatus.FAILED,
                errors=[f"Unsupported file format: {file_path}"]
            )
        
        with open(file_path, "rb") as f:
            return self.ingest(f, doc_format, doc_id, config, metadata)
    
    def _detect_format(self, file_path: str) -> Optional[DocumentFormat]:
        ext = os.path.splitext(file_path)[1].lower()
        
        word_formats = {".docx", ".doc"}
        image_formats = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        pdf_formats = {".pdf"}
        html_formats = {".html", ".htm"}
        
        if ext in word_formats:
            return DocumentFormat.WORD
        elif ext in pdf_formats:
            return DocumentFormat.PDF
        elif ext in image_formats:
            return DocumentFormat.IMAGE
        elif ext in html_formats:
            return DocumentFormat.HTML
        else:
            return None
    
    def supports_format(self, doc_format: DocumentFormat) -> bool:
        return doc_format in [
            DocumentFormat.WORD,
            DocumentFormat.PDF,
            DocumentFormat.IMAGE,
            DocumentFormat.HTML,
        ]
    
    def get_processing_mode(self, doc_format: DocumentFormat) -> str:
        """Return processing mode."""
        if doc_format == DocumentFormat.WORD:
            return "word_structured"
        else:
            return "layout_ocr"


class SimplifiedIngestPipeline:
    """Simplified ingest pipeline."""
    
    def __init__(
        self,
        ingest_adapter: SimplifiedIngestAdapter,
        cleaner_adapter: Optional[Any] = None,
        privacy_port: Optional[Any] = None
    ):
        self._logger = logging.getLogger("ingest.pipeline")
        self._adapter = ingest_adapter
        self._cleaner = cleaner_adapter
        self._privacy = privacy_port
    
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
        
        result = self._adapter.ingest(source, doc_format, doc_id, config, metadata)
        
        if result.status == IngestStatus.FAILED:
            return result
        
        if self._cleaner and result.content:
            from core.ports.cleaner import DocumentType
            
            cleaner_type_map = {
                DocumentFormat.PDF: DocumentType.PDF,
                DocumentFormat.WORD: DocumentType.WORD,
                DocumentFormat.HTML: DocumentType.HTML,
                DocumentFormat.IMAGE: DocumentType.TEXT,
            }
            
            cleaner_doc_type = cleaner_type_map.get(doc_format, DocumentType.TEXT)
            
            result.content = self._cleaner.clean(
                result.content,
                doc_type=cleaner_doc_type
            )
            
            for page in result.pages:
                page["content"] = self._cleaner.clean(
                    page["content"],
                    doc_type=cleaner_doc_type
                )
        
        if self._privacy and result.content:
            try:
                sensitivity = self._privacy.classify_sensitivity(result.content)
                result.metadata["sensitivity"] = sensitivity
                
                if sensitivity in ("high", "critical"):
                    self._logger.warning(
                        f"Document {doc_id} has high sensitivity: {sensitivity}"
                    )
            except (RuntimeError, ValueError) as e:
                self._logger.warning(f"Sensitivity classification failed: {e}")
        
        result.metadata["char_count"] = len(result.content)
        result.metadata["word_count"] = len(result.content.split())
        
        return result
    
    def ingest_from_path(
        self,
        file_path: str,
        doc_id: str,
        config: Optional[IngestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestResult:
        metadata = metadata or {}
        metadata["source_path"] = file_path
        
        doc_format = self._adapter._detect_format(file_path)
        
        if doc_format is None:
            return IngestResult(
                content="",
                metadata=metadata,
                status=IngestStatus.FAILED,
                errors=[f"Unsupported file format: {file_path}"]
            )
        
        with open(file_path, "rb") as f:
            return self.ingest(f, doc_format, doc_id, config, metadata)


def build_simplified_ingest_adapter(
    use_gpu: bool = False,
    ocr_backend: str = "auto",
    language: str = "ch"
) -> SimplifiedIngestAdapter:
    """Build simplified ingest adapter."""
    return SimplifiedIngestAdapter(
        use_gpu=use_gpu,
        ocr_backend=ocr_backend,
        language=language
    )


def build_simplified_ingest_pipeline(
    cleaner_adapter: Optional[Any] = None,
    privacy_port: Optional[Any] = None,
    use_gpu: bool = False,
    ocr_backend: str = "auto",
    language: str = "ch"
) -> SimplifiedIngestPipeline:
    """Build simplified ingest pipeline."""
    adapter = build_simplified_ingest_adapter(
        use_gpu=use_gpu,
        ocr_backend=ocr_backend,
        language=language
    )
    
    return SimplifiedIngestPipeline(
        ingest_adapter=adapter,
        cleaner_adapter=cleaner_adapter,
        privacy_port=privacy_port
    )
