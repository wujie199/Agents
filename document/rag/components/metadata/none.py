"""不打标（占位实现）。"""

from typing import Any, Dict, List, Optional

from core.ports.ingest import IngestResult


class NoOpMetadataEnricher:
    def enrich(
        self,
        ingest_result: IngestResult,
        *,
        doc_format: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        return ingest_result

    def enrich_batch(
        self,
        items: List[IngestResult],
        **kwargs: Any,
    ) -> List[IngestResult]:
        return items
