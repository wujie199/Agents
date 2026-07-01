"""摄取后 metadata 打标编排（清理之后、切块之前）。"""

import logging
from typing import Any, Optional

from core.ports.ingest import DocumentFormat, IngestResult, IngestStatus
from document.rag.config import RagPipelineConfig

logger = logging.getLogger("document.rag.application.metadata.pipeline")


def apply_metadata_enrichment(
    ingest_result: IngestResult,
    doc_format: DocumentFormat,
    cfg: RagPipelineConfig,
    enricher: Optional[Any] = None,
) -> IngestResult:
    """
    对 ingest 结果做文档级 metadata 打标。
    未启用或 ingest 失败时原样返回。
    """
    if ingest_result.status == IngestStatus.FAILED:
        return ingest_result

    meta_cfg = cfg.metadata
    if not meta_cfg.enabled:
        return ingest_result

    if enricher is None:
        from document.rag.components.metadata.registry import build_metadata_enricher

        enricher = build_metadata_enricher(cfg)

    return enricher.enrich(
        ingest_result,
        doc_format=doc_format.value,
    )
