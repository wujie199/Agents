"""摄取管道 — 从 application 层 re-export。"""

from document.rag.application.ingest_factory import (
    build_ingest_pipeline,
    build_routed_ingest,
    detect_format,
    RoutedIngestAdapter,
)

__all__ = ["build_ingest_pipeline", "build_routed_ingest", "detect_format", "RoutedIngestAdapter"]
