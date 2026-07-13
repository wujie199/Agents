import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.ports.chunker import ChunkStrategy, Chunk
from core.ports.index import IndexProfile, IndexResult
from core.ports.ingest import IngestResult, IngestStatus
from core.ports.rag.embedding import EmbeddingPort
from core.ports.storage.cache import CachePort
from core.ports.storage.vector import VectorPort, VectorRecord

from document.rag.config import RagPipelineConfig
from document.rag.application.indexing.chunker import create_chunker, parse_chunk_strategy
from document.rag.application.indexing.document_store import RagDocumentStore
from document.rag.application.embedding.collection import effective_collection_name
from document.rag.application.embedding.dlq import append_embedding_dlq
from document.rag.application.indexing.embedder import Embedder
from document.rag.application.indexing.graph_sync import RagGraphSync
from document.rag.application.chunking.chunker import SevenStepChunker
from document.rag.application.chunking.parent_store import ParentChunkStore
from document.rag.shared.dedupe import dedupe_chunks, semantic_dedupe


class IndexService:
    """Vector index: chunk, embed (ModelPort), write VectorPort; implements IndexPort."""

    def __init__(
        self,
        vector_port: VectorPort,
        embedding_model: EmbeddingPort,
        config: RagPipelineConfig,
        cache_port: Optional[CachePort] = None,
        chunk_strategy: Optional[ChunkStrategy] = None,
        sql_port: Optional[Any] = None,
        graph_port: Optional[Any] = None,
        bm25_index: Optional[Any] = None,
    ):
        self._config = config
        self._vector_port = vector_port
        self._embedding_model = embedding_model
        self._cache_port = cache_port
        self._collection = effective_collection_name(self._config)
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
        self._strategy = chunk_strategy or parse_chunk_strategy(self._config.chunk_strategy)

        embed_fn = self._build_embed_fn(embedding_model)
        chunker_kwargs: Dict[str, Any] = {
            "chunk_size": self._config.chunk_size,
            "chunk_overlap": self._config.chunk_overlap,
        }
        if self._strategy == ChunkStrategy.SEVEN_STEP:
            chunker_kwargs["pipeline_cfg"] = self._config.chunk_pipeline
            chunker_kwargs["embed_fn"] = embed_fn

        self._chunker = create_chunker(self._strategy, **chunker_kwargs)
        parent_dir = getattr(self._config.chunk_pipeline, "parent_store_dir", "parent_chunks")
        self._parent_store = ParentChunkStore(Path(parent_dir))
        self._embedder = Embedder(
            embedding_model=embedding_model,
            embedding_cfg=config.embedding,
            cache_port=cache_port,
            model_version=self._config.model_version,
            enable_cache=cache_port is not None,
        )

    @property
    def collection_name(self) -> str:
        return self._collection

    def _build_embed_fn(self, embedding_model: EmbeddingPort):
        if not hasattr(embedding_model, "embed"):
            return None

        def _embed(texts: List[str]) -> List[List[float]]:
            return embedding_model.embed(texts)

        return _embed

    def _uses_seven_step_pipeline(self) -> bool:
        return self._strategy == ChunkStrategy.SEVEN_STEP

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

    def _dedupe_chunk_list(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return chunks
        if self._uses_seven_step_pipeline():
            for idx, chunk in enumerate(chunks):
                chunk.chunk_index = idx
            return chunks

        if self._config.enable_chunk_dedupe:
            payload = [
                {"chunk_text": c.content, "_chunk": c} for c in chunks
            ]
            payload = dedupe_chunks(payload)
            chunks = [item["_chunk"] for item in payload]

        if self._config.enable_semantic_dedupe:
            payload = [
                {"chunk_text": c.content, "_chunk": c} for c in chunks
            ]
            payload = semantic_dedupe(
                payload,
                threshold=self._config.semantic_dedupe_threshold,
            )
            chunks = [item["_chunk"] for item in payload]

        for idx, chunk in enumerate(chunks):
            chunk.chunk_index = idx
        return chunks

    async def index_document(
        self,
        doc_id: str,
        content: str,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        profile: Optional[IndexProfile] = None,
        previous_fingerprints: Optional[Dict[str, str]] = None,
    ) -> IndexResult:
        resolved_profile = self._resolve_profile(profile)
        metadata = dict(metadata or {})
        metadata["tenant_id"] = tenant_id
        metadata["doc_id"] = doc_id

        chunks = self._chunker.chunk(content, doc_id, metadata)
        if isinstance(self._chunker, SevenStepChunker):
            parents = self._chunker.get_parent_chunks()
            if parents:
                self._parent_store.save(self._collection, doc_id, parents)
            stats = self._chunker.get_pipeline_stats()
            metadata["chunk_pipeline_stats"] = stats
        before = len(chunks)
        chunks = self._dedupe_chunk_list(chunks)
        if len(chunks) != before:
            self._logger.info(
                "Document %s deduped chunks %d -> %d",
                doc_id,
                before,
                len(chunks),
            )
        self._logger.info("Document %s split into %d chunks", doc_id, len(chunks))

        embed_result = await self._embedder.embed_chunks(
            chunks,
            tenant_id,
            previous_fingerprints=previous_fingerprints,
        )
        chunks_deleted = 0
        if self._config.embedding.enable_chunk_incremental:
            chunks_deleted = await self._delete_orphan_chunks(
                doc_id,
                tenant_id,
                {chunk.chunk_id for chunk in chunks},
            )
        if embed_result.skipped_unchanged or embed_result.encoded_count:
            self._logger.info(
                "Document %s embed: encoded=%d skipped_unchanged=%d deleted=%d total=%d",
                doc_id,
                embed_result.encoded_count,
                embed_result.skipped_unchanged,
                chunks_deleted,
                len(chunks),
            )

        bm25_written = False
        bm25_needs_update = (
            embed_result.encoded_count > 0
            or chunks_deleted > 0
            or not self._config.embedding.enable_chunk_incremental
        )
        if self._bm25_index is not None and bm25_needs_update:
            bm25_payload = [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": {**chunk.metadata, "tenant_id": tenant_id},
                }
                for chunk in chunks
            ]
            await asyncio.to_thread(
                self._bm25_index.index_chunks,
                bm25_payload,
                tenant_id,
                doc_id,
            )
            bm25_written = True
        try:
            written = await self._write_vectors(
                embed_result.to_write,
                doc_id=doc_id,
                tenant_id=tenant_id,
            )
        except (RuntimeError, ConnectionError, OSError, ValueError):
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
            chunk_fingerprints=embed_result.fingerprints,
            chunks_skipped_unchanged=embed_result.skipped_unchanged,
            chunks_deleted=chunks_deleted,
            chunks_encoded=embed_result.encoded_count,
        )

    async def index_from_ingest(
        self,
        ingest_result: IngestResult,
        tenant_id: str,
        doc_id: Optional[str] = None,
        profile: Optional[IndexProfile] = None,
        previous_fingerprints: Optional[Dict[str, str]] = None,
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
            previous_fingerprints=previous_fingerprints,
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
            self._parent_store.delete(self._collection, doc_id)
            self._logger.info(
                "Deleted document %s from %s (%d chunks)",
                doc_id,
                self._collection,
                deleted,
            )
            return deleted > 0
        except (RuntimeError, ConnectionError, OSError, ValueError) as e:
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
        except (ConnectionError, TimeoutError, OSError, RuntimeError):
            stats["vector_count"] = "unknown"

        if self._cache_port:
            version_key = f"{tenant_id}:rag:meta:index_version"
            try:
                version = await self._cache_get(version_key)
                stats["index_version"] = version or 0
            except (ConnectionError, TimeoutError, OSError, ValueError, KeyError):
                stats["index_version"] = "unknown"

        return stats

    async def _delete_orphan_chunks(
        self,
        doc_id: str,
        tenant_id: str,
        current_chunk_ids: set[str],
    ) -> int:
        list_fn = getattr(self._vector_port, "list_ids_by_filter", None)
        delete_fn = getattr(self._vector_port, "delete_by_ids", None)
        if not list_fn or not delete_fn:
            return 0
        try:
            existing = await asyncio.to_thread(
                list_fn,
                self._collection,
                {"doc_id": doc_id, "tenant_id": tenant_id},
            )
        except (RuntimeError, ConnectionError, OSError, ValueError) as exc:
            self._logger.warning("列举 chunk id 失败 doc=%s: %s", doc_id, exc)
            return 0
        orphan_ids = [cid for cid in existing if cid not in current_chunk_ids]
        if not orphan_ids:
            return 0
        try:
            deleted = await asyncio.to_thread(
                delete_fn,
                self._collection,
                orphan_ids,
            )
            self._logger.info(
                "Deleted %d orphan chunks for doc=%s", deleted, doc_id
            )
            return int(deleted)
        except (RuntimeError, ConnectionError, OSError, ValueError) as exc:
            self._logger.warning("删除 orphan chunk 失败 doc=%s: %s", doc_id, exc)
            return 0

    async def _write_vectors(
        self,
        embeddings: List[Dict[str, Any]],
        *,
        doc_id: str = "",
        tenant_id: str = "",
    ) -> int:
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
        expected = len(records)
        max_retries = max(1, int(self._config.embedding.write_max_retries))
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                written = await asyncio.to_thread(
                    self._vector_port.upsert,
                    self._collection,
                    records,
                )
                if written != expected:
                    raise RuntimeError(
                        f"向量写入数量不一致: expected={expected} written={written}"
                    )
                return written
            except (RuntimeError, ConnectionError, OSError, ValueError) as exc:
                last_exc = exc
                self._logger.warning(
                    "向量写入失败 (%d/%d): %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )
                if attempt + 1 < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
        assert last_exc is not None
        append_embedding_dlq(
            self._config.embedding.dlq_path,
            collection=self._collection,
            doc_id=doc_id,
            tenant_id=tenant_id,
            expected=expected,
            error=str(last_exc),
        )
        raise last_exc

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
