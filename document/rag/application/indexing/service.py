import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.ports.chunker import ChunkStrategy
from core.ports.index import IndexProfile, IndexResult
from core.ports.ingest import IngestResult, IngestStatus
from core.ports.storage.cache import CachePort
from core.ports.storage.vector import VectorPort, VectorRecord

from document.rag.config import RagPipelineConfig, load_rag_pipeline_config
from document.rag.application.indexing.chunker import create_chunker, parse_chunk_strategy
from document.rag.application.indexing.document_store import RagDocumentStore
from document.rag.application.indexing.embedder import Embedder
from document.rag.application.indexing.graph_sync import RagGraphSync


class IndexService:
    """Vector index: chunk, embed (ModelPort), write VectorPort; implements IndexPort."""

    def __init__(
        self,
        vector_port: VectorPort,
        embedding_model: Any,
        config: Optional[RagPipelineConfig] = None,
        cache_port: Optional[CachePort] = None,
        chunk_strategy: Optional[ChunkStrategy] = None,
        sql_port: Optional[Any] = None,
        graph_port: Optional[Any] = None,
        bm25_index: Optional[Any] = None,
    ):
        self._config = config or load_rag_pipeline_config()
        self._vector_port = vector_port
        self._embedding_model = embedding_model
        self._cache_port = cache_port
        self._collection = self._config.collection_name
        self._sql_port = sql_port
        self._graph_port = graph_port
        self._doc_store = (
            RagDocumentStore(sql_port) if sql_port and self._config.retrieval.enable_sql else None
        )
        self._graph_sync = (
            RagGraphSync(graph_port)
            if graph_port and self._config.enable_graph_index
            else None
        )
        self._bm25_index = bm25_index
        self._logger = logging.getLogger("knowledge.pipeline.index.service")

        self._chunker = create_chunker(
            chunk_strategy or parse_chunk_strategy(self._config.chunk_strategy),
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        self._embedder = Embedder(
            embedding_model=embedding_model,
            batch_size=self._config.embedding_batch_size,
            cache_port=cache_port,
            model_version=self._config.model_version,
            enable_cache=cache_port is not None,
        )

    @property
    def collection_name(self) -> str:
        return self._collection

    def _resolve_profile(self, profile: Optional[IndexProfile]) -> IndexProfile:
        if profile is not None:
            return profile
        if self._doc_store and self._graph_sync:
            return IndexProfile.FULL
        if self._doc_store:
            return IndexProfile.SQL_SIDECAR
        if self._graph_sync:
            return IndexProfile.GRAPH_SIDECAR
        return IndexProfile.VECTOR_ONLY

    async def index_document(
        self,
        doc_id: str,
        content: str,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        profile: Optional[IndexProfile] = None,
    ) -> IndexResult:
        resolved_profile = self._resolve_profile(profile)
        metadata = dict(metadata or {})
        metadata["tenant_id"] = tenant_id
        metadata["doc_id"] = doc_id

        chunks = self._chunker.chunk(content, doc_id, metadata)
        self._logger.info("Document %s split into %d chunks", doc_id, len(chunks))

        embeddings = await self._embedder.embed_chunks(chunks, tenant_id)
        bm25_written = False
        if self._bm25_index is not None:
            await asyncio.to_thread(
                self._bm25_index.index_chunks,
                embeddings,
                tenant_id,
                doc_id,
            )
            bm25_written = True
        try:
            written = await self._write_vectors(embeddings)
        except Exception:
            if bm25_written and self._bm25_index is not None:
                await asyncio.to_thread(
                    self._bm25_index.delete_by_doc_id,
                    doc_id,
                    tenant_id,
                )
            raise
        side_indexes = await self._sync_side_indexes(
            doc_id, tenant_id, content, metadata, resolved_profile
        )
        index_version = await self._bump_index_version(tenant_id)

        return IndexResult(
            doc_id=doc_id,
            chunk_count=len(chunks),
            vectors_written=written,
            collection=self._collection,
            indexed_at=datetime.utcnow().isoformat(),
            model_version=self._config.model_version,
            profile=resolved_profile,
            side_indexes=side_indexes,
            index_version=index_version,
        )

    async def index_from_ingest(
        self,
        ingest_result: IngestResult,
        tenant_id: str,
        doc_id: Optional[str] = None,
        profile: Optional[IndexProfile] = None,
    ) -> IndexResult:
        if ingest_result.status == IngestStatus.FAILED:
            raise ValueError(f"Cannot index failed ingest: {ingest_result.errors}")

        doc_id = doc_id or ingest_result.metadata.get("doc_id")
        if not doc_id:
            raise ValueError("doc_id is required for indexing")

        metadata = dict(ingest_result.metadata)
        metadata.setdefault("source_path", metadata.get("source_path"))
        return await self.index_document(
            doc_id=doc_id,
            content=ingest_result.content,
            tenant_id=tenant_id,
            metadata=metadata,
            profile=profile,
        )

    async def index_documents_batch(
        self,
        docs: List[Dict[str, Any]],
        tenant_id: str,
    ) -> List[IndexResult]:
        results = []
        for doc in docs:
            result = await self.index_document(
                doc_id=doc["doc_id"],
                content=doc["content"],
                tenant_id=tenant_id,
                metadata=doc.get("metadata"),
                profile=doc.get("profile"),
            )
            results.append(result)
        return results

    async def delete_document(self, doc_id: str, tenant_id: str) -> bool:
        try:
            deleted = await asyncio.to_thread(
                self._vector_port.delete_by_filter,
                self._collection,
                {"doc_id": doc_id, "tenant_id": tenant_id},
            )
            if self._doc_store:
                await self._doc_store.delete_document(doc_id, tenant_id)
            if self._graph_sync:
                self._graph_sync.delete_document(doc_id)
            if self._bm25_index is not None:
                await asyncio.to_thread(
                    self._bm25_index.delete_by_doc_id,
                    doc_id,
                    tenant_id,
                )
            self._logger.info(
                "Deleted document %s from %s (%d chunks)",
                doc_id,
                self._collection,
                deleted,
            )
            return deleted > 0
        except Exception as e:
            self._logger.error("Failed to delete document %s: %s", doc_id, e)
            return False

    async def _sync_side_indexes(
        self,
        doc_id: str,
        tenant_id: str,
        content: str,
        metadata: Dict[str, Any],
        profile: IndexProfile,
    ) -> Dict[str, bool]:
        side: Dict[str, bool] = {"sql": False, "graph": False}
        if profile in (IndexProfile.SQL_SIDECAR, IndexProfile.FULL) and self._doc_store:
            await self._doc_store.upsert_document(
                doc_id=doc_id,
                tenant_id=tenant_id,
                content=content,
                metadata=metadata,
            )
            side["sql"] = True
        if profile in (IndexProfile.GRAPH_SIDECAR, IndexProfile.FULL) and self._graph_sync:
            await asyncio.to_thread(
                self._graph_sync.upsert_document,
                doc_id,
                tenant_id,
                content,
                metadata,
            )
            side["graph"] = True
        return side

    async def get_index_stats(self, tenant_id: str) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "collection": self._collection,
            "model_version": self._config.model_version,
            "tenant_id": tenant_id,
        }
        try:
            stats["vector_count"] = await asyncio.to_thread(
                self._vector_port.count,
                self._collection,
            )
        except Exception:
            stats["vector_count"] = "unknown"

        if self._cache_port:
            version_key = f"{tenant_id}:rag:meta:index_version"
            try:
                version = await self._cache_get(version_key)
                stats["index_version"] = version or 0
            except Exception:
                stats["index_version"] = "unknown"

        return stats

    async def _write_vectors(self, embeddings: List[Dict[str, Any]]) -> int:
        records = [
            VectorRecord(
                id=item["chunk_id"],
                vector=item["embedding"],
                metadata=item["metadata"],
                content=item["content"],
            )
            for item in embeddings
        ]
        if not records:
            return 0
        return await asyncio.to_thread(
            self._vector_port.upsert,
            self._collection,
            records,
        )

    async def _bump_index_version(self, tenant_id: str) -> Optional[int]:
        if self._cache_port is None:
            return None
        version_key = f"{tenant_id}:rag:meta:index_version"
        current = await self._cache_get(version_key)
        new_version = (current or 0) + 1
        await self._cache_set(version_key, new_version)
        return new_version

    async def _cache_get(self, key: str) -> Any:
        value = self._cache_port.get(key)
        if asyncio.iscoroutine(value):
            return await value
        return value

    async def _cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        result = self._cache_port.set(key, value, ttl)
        if asyncio.iscoroutine(result):
            await result
