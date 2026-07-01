"""知识库端口。"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from core.ports.index import IndexProfile, IndexResult
from core.ports.rag.ingest import IngestConfig, IngestResult


@dataclass
class IngestAndIndexResult:
    success: bool
    doc_id: str
    tenant_id: str
    ingest: Optional[IngestResult] = None
    index: Optional[IndexResult] = None
    index_profile: IndexProfile = IndexProfile.VECTOR_ONLY
    errors: List[str] = field(default_factory=list)


class KnowledgeBasePort(Protocol):
    """建库门面：文件摄取 → 可选脱敏 → IndexPort 入库。"""

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
        ...

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
        ...
