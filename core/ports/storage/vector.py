from typing import Protocol, List, Optional
from dataclasses import dataclass


@dataclass
class VectorRecord:
    id: str
    vector: List[float]
    metadata: dict
    content: Optional[str] = None


@dataclass
class SearchResult:
    id: str
    score: float
    content: Optional[str] = None
    metadata: Optional[dict] = None


class VectorPort(Protocol):
    def upsert(
        self,
        collection: str,
        records: List[VectorRecord]
    ) -> int:
        ...

    def similarity_search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[dict] = None
    ) -> List[SearchResult]:
        ...

    def delete_by_doc_id(
        self,
        collection: str,
        doc_id: str
    ) -> int:
        ...

    def delete_by_ids(
        self,
        collection: str,
        ids: List[str]
    ) -> int:
        ...

    def list_ids_by_filter(
        self,
        collection: str,
        filter: dict,
    ) -> List[str]:
        ...

    def get_index_version(self, collection: str) -> str:
        ...
