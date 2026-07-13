import asyncio
import hashlib
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.ports.chunker import Chunk
from core.ports.rag.embedding import EmbeddingPort
from core.ports.storage.cache import CachePort
from document.rag.application.embedding.encoder import EmbeddingEncoder
from document.rag.application.embedding.fingerprint import chunk_embed_fingerprint
from document.rag.application.embedding.text_prep import prepare_texts
from document.rag.config.embedding import EmbeddingConfig


@dataclass
class EmbedChunksResult:
    """向量化结果：仅 to_write 需 upsert 向量库。"""

    to_write: List[Dict[str, Any]] = field(default_factory=list)
    fingerprints: Dict[str, str] = field(default_factory=dict)
    skipped_unchanged: int = 0
    encoded_count: int = 0


class Embedder:
    """Chunk 向量化：五步流水线 Step1–3 + 缓存读/写 + chunk 增量跳过重编码。"""

    def __init__(
        self,
        embedding_model: EmbeddingPort,
        embedding_cfg: Optional[EmbeddingConfig] = None,
        batch_size: Optional[int] = None,
        cache_port: Optional[CachePort] = None,
        model_version: str = "v1",
        enable_cache: bool = True,
    ):
        cfg = embedding_cfg or EmbeddingConfig()
        if batch_size is not None:
            cfg = replace(cfg, batch_size=int(batch_size))
        self._encoder = EmbeddingEncoder(embedding_model, cfg)
        self._cfg = cfg
        self._cache_port = cache_port
        self._model_version = model_version
        self._enable_cache = enable_cache
        self._logger = logging.getLogger("rag.embedder")

    async def embed_chunks(
        self,
        chunks: List[Chunk],
        tenant_id: str,
        *,
        previous_fingerprints: Optional[Dict[str, str]] = None,
    ) -> EmbedChunksResult:
        if not chunks:
            return EmbedChunksResult()

        prev = previous_fingerprints or {}
        use_incremental = bool(self._cfg.enable_chunk_incremental and prev)
        cache_read = bool(
            self._enable_cache
            and self._cache_port
            and self._cfg.enable_embedding_cache_read
        )

        texts = [
            chunk.metadata.get("_embed_content") or chunk.content
            for chunk in chunks
        ]
        prepared = prepare_texts(
            texts,
            self._cfg,
            "doc",
            tokenize_fn=self._encoder._tokenize_fn,
            decode_truncate_fn=self._encoder._decode_truncate_fn,
        )
        prep_by_index = {item.original_index: item for item in prepared.items}

        result = EmbedChunksResult()
        pending_indices: List[int] = []
        pending_texts: List[str] = []

        for idx, chunk in enumerate(chunks):
            item = prep_by_index.get(idx)
            if item is None:
                for skip_idx, reason in prepared.skipped:
                    if skip_idx == idx:
                        self._logger.warning(
                            "跳过 chunk index=%s doc=%s 原因=%s",
                            idx,
                            chunk.doc_id,
                            reason,
                        )
                continue

            fp = chunk_embed_fingerprint(item.text, model_version=self._model_version)
            result.fingerprints[chunk.chunk_id] = fp

            if use_incremental and prev.get(chunk.chunk_id) == fp:
                result.skipped_unchanged += 1
                continue

            if cache_read:
                cached = await self._cache_get_embedding(item.text)
                if cached is not None:
                    result.to_write.append(
                        self._build_record(chunk, cached, tenant_id)
                    )
                    await self._maybe_cache_embedding(item.text, cached)
                    continue

            pending_indices.append(idx)
            pending_texts.append(item.text)

        if pending_texts:
            vectors, batch_prepared = await self._encoder.encode_texts(
                pending_texts, "doc"
            )
            if batch_prepared.skipped:
                for skip_idx, reason in batch_prepared.skipped:
                    orig = pending_indices[skip_idx] if skip_idx < len(pending_indices) else "?"
                    self._logger.warning(
                        "编码跳过 pending_index=%s 原因=%s", orig, reason
                    )
            if len(vectors) != len(batch_prepared.items):
                raise RuntimeError("向量化结果与待编码 chunk 数量不一致")

            for item, embedding in zip(batch_prepared.items, vectors):
                chunk_idx = pending_indices[item.original_index]
                chunk = chunks[chunk_idx]
                prepared_text = item.text
                await self._maybe_cache_embedding(prepared_text, embedding)
                result.to_write.append(
                    self._build_record(chunk, embedding, tenant_id)
                )
                result.encoded_count += 1

        return result

    def _build_record(
        self,
        chunk: Chunk,
        embedding: List[float],
        tenant_id: str,
    ) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "content": chunk.content,
            "embedding": embedding,
            "metadata": {
                **chunk.metadata,
                "tenant_id": tenant_id,
                "indexed_at": datetime.utcnow().isoformat(),
                "model_version": self._model_version,
            },
        }

    async def _cache_get_embedding(self, prepared_text: str) -> Optional[List[float]]:
        if not self._cache_port:
            return None
        cache_key = self._get_embedding_cache_key(prepared_text)
        try:
            cached = await self._cache_get(cache_key)
        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
            self._logger.warning("Cache get failed: %s", exc)
            return None
        if not cached:
            return None
        embedding = cached.get("embedding") if isinstance(cached, dict) else cached
        if not isinstance(embedding, list) or not embedding:
            return None
        from document.rag.application.embedding.normalizer import normalize_vector

        normalized = normalize_vector(embedding, self._cfg)
        if normalized is None:
            return None
        return normalized

    async def _maybe_cache_embedding(
        self, prepared_text: str, embedding: List[float]
    ) -> None:
        if not self._enable_cache or not self._cache_port:
            return
        cache_key = self._get_embedding_cache_key(prepared_text)
        try:
            await self._cache_set(
                cache_key,
                {"embedding": embedding},
                ttl_seconds=86400 * 30,
            )
        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
            self._logger.warning("Cache set failed: %s", exc)

    def _get_embedding_cache_key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"emb:{self._model_version}:{text_hash}"

    async def _cache_get(self, key: str) -> Any:
        if not self._cache_port:
            return None
        value = self._cache_port.get(key)
        if asyncio.iscoroutine(value):
            return await value
        return value

    async def _cache_set(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> None:
        if not self._cache_port:
            return
        result = self._cache_port.set(key, value, ttl_seconds)
        if asyncio.iscoroutine(result):
            await result
