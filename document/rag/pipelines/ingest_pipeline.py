"""摄取管道 re-export。"""

from document.rag.application.ingest_factory import (
    build_ingest_pipeline,
    detect_format,
)

__all__ = ["build_ingest_pipeline", "detect_format"]
