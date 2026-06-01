from typing import List, Optional
import chromadb
from chromadb.config import Settings
from core.ports.storage.vector import VectorPort, VectorRecord, SearchResult
from document.rag.application.retrieval.tag_filter import chroma_safe_metadata


def _normalize_where(filter: Optional[dict]) -> Optional[dict]:
    if not filter:
        return filter
    if len(filter) <= 1:
        return filter
    return {"$and": [{key: value} for key, value in filter.items()]}


class ChromaVectorAdapter:
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        if host and port:
            self._client = chromadb.HttpClient(host=host, port=port)
        elif persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()
        
        self._collections: dict = {}
    
    def _get_collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(name)
        return self._collections[name]
    
    def upsert(
        self,
        collection: str,
        records: List[VectorRecord]
    ) -> int:
        col = self._get_collection(collection)
        
        ids = [r.id for r in records]
        embeddings = [r.vector for r in records]
        metadatas = [chroma_safe_metadata(r.metadata) for r in records]
        documents = [r.content for r in records]
        
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        return len(records)
    
    def similarity_search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[dict] = None
    ) -> List[SearchResult]:
        col = self._get_collection(collection)
        
        results = col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=_normalize_where(filter),
        )
        
        search_results = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        for i, id in enumerate(ids):
            score = 1 - distances[i] if distances else 0.0
            search_results.append(SearchResult(
                id=id,
                score=score,
                content=documents[i] if documents else None,
                metadata=metadatas[i] if metadatas else None
            ))
        
        return search_results
    
    def delete_by_doc_id(
        self,
        collection: str,
        doc_id: str,
        tenant_id: Optional[str] = None,
    ) -> int:
        col = self._get_collection(collection)

        where = {"doc_id": doc_id}
        if tenant_id:
            where = {"doc_id": doc_id, "tenant_id": tenant_id}
        results = col.get(where=_normalize_where(where))

        if results and results.get("ids"):
            col.delete(ids=results["ids"])
            return len(results["ids"])

        return 0

    def delete_by_filter(self, collection: str, filter: dict) -> int:
        col = self._get_collection(collection)
        results = col.get(where=_normalize_where(filter))
        if results and results.get("ids"):
            col.delete(ids=results["ids"])
            return len(results["ids"])
        return 0
    
    def delete_by_ids(
        self,
        collection: str,
        ids: List[str]
    ) -> int:
        col = self._get_collection(collection)
        col.delete(ids=ids)
        return len(ids)
    
    def get_index_version(self, collection: str) -> str:
        col = self._get_collection(collection)
        metadata = col.metadata or {}
        return metadata.get("index_version", "v1")
    
    def set_index_version(self, collection: str, version: str) -> None:
        col = self._get_collection(collection)
        col.modify(metadata={"index_version": version})
    
    def get_by_ids(
        self,
        collection: str,
        ids: List[str]
    ) -> List[SearchResult]:
        col = self._get_collection(collection)
        
        results = col.get(ids=ids)
        
        search_results = []
        ids_list = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        
        for i, id in enumerate(ids_list):
            search_results.append(SearchResult(
                id=id,
                score=1.0,
                content=documents[i] if documents else None,
                metadata=metadatas[i] if metadatas else None
            ))
        
        return search_results
    
    def count(self, collection: str) -> int:
        col = self._get_collection(collection)
        return col.count()
    
    def health(self) -> dict:
        try:
            self._client.heartbeat()
            return {"status": "healthy", "type": "chroma"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
