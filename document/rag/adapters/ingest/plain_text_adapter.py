import io
import logging
from typing import Any, BinaryIO, Dict, Optional

from core.ports.ingest import (
    DocumentFormat,
    IngestConfig,
    IngestPort,
    IngestResult,
    IngestStatus,
)


class PlainTextIngestAdapter:
    """Module docstring."""

    def __init__(self, encoding: str = "utf-8"):
        self._encoding = encoding
        self._logger = logging.getLogger("ingest.plain_text")

    def supports_format(self, doc_format: DocumentFormat) -> bool:
        return doc_format in (DocumentFormat.TEXT, DocumentFormat.MARKDOWN)

    def ingest(
        self,
        source: BinaryIO,
        doc_format: DocumentFormat,
        doc_id: str,
        config: Optional[IngestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        metadata = dict(metadata or {})
        metadata["doc_id"] = doc_id
        metadata["doc_format"] = doc_format.value

        try:
            raw = source.read()
            if isinstance(raw, bytes):
                content = raw.decode(self._encoding, errors="replace")
            else:
                content = str(raw)
        except Exception as e:
            self._logger.error("Failed to read plain text %s: %s", doc_id, e)
            return IngestResult(
                content="",
                metadata=metadata,
                status=IngestStatus.FAILED,
                errors=[str(e)],
            )

        return IngestResult(
            content=content.strip(),
            metadata=metadata,
            status=IngestStatus.SUCCESS if content.strip() else IngestStatus.PARTIAL,
        )

    def ingest_from_path(
        self,
        file_path: str,
        doc_id: str,
        config: Optional[IngestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        metadata = dict(metadata or {})
        metadata["source_path"] = file_path
        with open(file_path, "rb") as f:
            ext = file_path.rsplit(".", 1)[-1].lower()
            fmt = (
                DocumentFormat.MARKDOWN
                if ext in ("md", "markdown")
                else DocumentFormat.TEXT
            )
            return self.ingest(f, fmt, doc_id, config, metadata)
