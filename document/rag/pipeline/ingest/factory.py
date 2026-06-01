from document.rag.application.ingest.factory import (
    RoutedIngestAdapter,
    build_ingest_pipeline,
    build_routed_ingest,
    detect_format,
)

__all__ = [
    "detect_format",
    "build_ingest_pipeline",
    "build_routed_ingest",
    "RoutedIngestAdapter",
]
