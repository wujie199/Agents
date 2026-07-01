"""Ingest 端口 — 从 core.ports 重新导出，保持单一来源。"""

from core.ports.rag.ingest import IngestPort, IngestResult, IngestConfig, IngestStatus, DocumentFormat

__all__ = ["IngestPort", "IngestResult", "IngestConfig", "IngestStatus", "DocumentFormat"]
