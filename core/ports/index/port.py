from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol

from core.ports.rag.ingest import IngestResult


class IndexProfile(str, Enum):
    """建库侧索引策略（与检索 RoutingPort 无关）。"""

    VECTOR_ONLY = "vector_only"
    SQL_SIDECAR = "sql_sidecar"
    GRAPH_SIDECAR = "graph_sidecar"
    FULL = "full"


@dataclass
class IndexResult:
    doc_id: str
    chunk_count: int
    vectors_written: int
    collection: str
    indexed_at: str
    model_version: str
    profile: IndexProfile = IndexProfile.VECTOR_ONLY
    side_indexes: Dict[str, bool] = field(default_factory=dict)
    index_version: Optional[Any] = None

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        profile: IndexProfile = IndexProfile.VECTOR_ONLY,
        side_indexes: Optional[Dict[str, bool]] = None,
    ) -> "IndexResult":
        return cls(
            doc_id=str(data["doc_id"]),
            chunk_count=int(data.get("chunk_count", 0)),
            vectors_written=int(data.get("vectors_written", 0)),
            collection=str(data.get("collection", "")),
            indexed_at=str(data.get("indexed_at", "")),
            model_version=str(data.get("model_version", "")),
            profile=profile,
            side_indexes=side_indexes or {},
            index_version=data.get("index_version"),
        )


class IndexPort(Protocol):
    """RAG 写入：分块 → 向量化（ModelPort）→ VectorPort，可选 SQL/图侧写。"""

    async def index_document(
        self,
        doc_id: str,
        content: str,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        profile: Optional[IndexProfile] = None,
    ) -> IndexResult:
        ...

    async def index_from_ingest(
        self,
        ingest_result: IngestResult,
        tenant_id: str,
        doc_id: Optional[str] = None,
        profile: Optional[IndexProfile] = None,
    ) -> IndexResult:
        ...

    async def delete_document(self, doc_id: str, tenant_id: str) -> bool:
        ...

    async def get_index_stats(self, tenant_id: str) -> Dict[str, Any]:
        ...
