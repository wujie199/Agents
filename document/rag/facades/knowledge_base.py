import logging
from typing import Any, Dict, Optional

from core.ports.cleaner import CleaningLevel
from core.ports.index import IndexPort, IndexProfile
from core.ports.ingest import IngestConfig, IngestPort, IngestStatus
from core.ports.knowledge_base import IngestAndIndexResult, KnowledgeBasePort
from core.ports.privacy import PrivacyPort
from document.rag.adapters.cleaning.factory import build_enterprise_cleaner
from document.rag.config import IngestConfig as PipelineIngestConfig, RagPipelineConfig
from document.rag.application.cleaning.pipeline import apply_ingest_cleaning, parse_cleaning_level
from document.rag.application.metadata.pipeline import apply_metadata_enrichment
from document.rag.application.ingest.factory import detect_format

__all__ = ["KnowledgeBasePortAdapter", "parse_cleaning_level"]


class KnowledgeBasePortAdapter:
    """Ingest + index write facade (KnowledgeBasePort)."""

    def __init__(
        self,
        ingest_port: IngestPort,
        index_port: IndexPort,
        privacy_port: Optional[PrivacyPort] = None,
        default_index_profile: Optional[IndexProfile] = None,
        enable_cleaning: bool = True,
        ocr_postprocess: bool = True,
        cleaning_level: CleaningLevel = CleaningLevel.STANDARD,
        cleaner: Optional[Any] = None,
        rag_config: Optional[RagPipelineConfig] = None,
    ):
        self._ingest = ingest_port
        self._index = index_port
        self._privacy = privacy_port
        self._default_profile = default_index_profile or IndexProfile.FULL
        self._cleaner = cleaner or build_enterprise_cleaner()
        self._logger = logging.getLogger("document.rag.knowledge_base")

        if rag_config is not None:
            self._rag_config = rag_config
        else:
            level = (
                cleaning_level.value
                if isinstance(cleaning_level, CleaningLevel)
                else str(cleaning_level)
            )
            self._rag_config = RagPipelineConfig(
                ingest=PipelineIngestConfig(
                    enable_cleaning=enable_cleaning,
                    ocr_postprocess=ocr_postprocess,
                    cleaning_level=level,
                )
            )

    async def ingest_and_index(
        self,
        file_path: str,
        doc_id: str,
        tenant_id: str,
        index_profile: Optional[IndexProfile] = None,
        ingest_config: Optional[IngestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        mask_pii: bool = False,
    ) -> IngestAndIndexResult:
        profile = index_profile or self._default_profile
        meta = dict(metadata or {})
        meta.setdefault("tenant_id", tenant_id)

        ingest_result = self._ingest.ingest_from_path(
            file_path, doc_id, ingest_config, meta
        )

        if ingest_result.status == IngestStatus.FAILED:
            return IngestAndIndexResult(
                success=False,
                doc_id=doc_id,
                tenant_id=tenant_id,
                ingest=ingest_result,
                index_profile=profile,
                errors=list(ingest_result.errors) or ["ingest failed"],
            )

        doc_format = detect_format(file_path)
        apply_ingest_cleaning(
            ingest_result, doc_format, self._rag_config, self._cleaner
        )
        apply_metadata_enrichment(ingest_result, doc_format, self._rag_config)

        content = ingest_result.content or ""
        if mask_pii and self._privacy and content:
            content = self._privacy.mask_text(content)
            ingest_result.content = content

        try:
            index_result = await self._index.index_from_ingest(
                ingest_result,
                tenant_id=tenant_id,
                doc_id=doc_id,
                profile=profile,
            )
        except Exception as exc:
            self._logger.error("Index after ingest failed: %s", exc)
            return IngestAndIndexResult(
                success=False,
                doc_id=doc_id,
                tenant_id=tenant_id,
                ingest=ingest_result,
                index_profile=profile,
                errors=[str(exc)],
            )

        return IngestAndIndexResult(
            success=True,
            doc_id=doc_id,
            tenant_id=tenant_id,
            ingest=ingest_result,
            index=index_result,
            index_profile=profile,
        )

    async def reindex_document(
        self,
        file_path: str,
        doc_id: str,
        tenant_id: str,
        index_profile: Optional[IndexProfile] = None,
        ingest_config: Optional[IngestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        mask_pii: bool = False,
    ) -> IngestAndIndexResult:
        await self._index.delete_document(doc_id, tenant_id)
        return await self.ingest_and_index(
            file_path=file_path,
            doc_id=doc_id,
            tenant_id=tenant_id,
            index_profile=index_profile,
            ingest_config=ingest_config,
            metadata=metadata,
            mask_pii=mask_pii,
        )
