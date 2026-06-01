import asyncio
import hashlib
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import logging

from core.ports.chunker import Chunk


class Embedder:
    """
    
    æ¯æï¼?    - æ¹é embedding
    - Embedding ç¼å­
    - å¤±è´¥éè¯
    """
    
    def __init__(
        self,
        embedding_model: Any,
        batch_size: int = 32,
        cache_port: Optional[Any] = None,
        model_version: str = "v1",
        enable_cache: bool = True,
    ):
        self._model = embedding_model
        self._batch_size = batch_size
        self._cache_port = cache_port
        self._model_version = model_version
        self._enable_cache = enable_cache
        self._logger = logging.getLogger("rag.embedder")
    
    async def embed_chunks(
        self,
        chunks: List[Chunk],
        tenant_id: str
    ) -> List[Dict[str, Any]]:
        results = []
        
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i:i + self._batch_size]
            batch_results = await self._embed_batch(batch, tenant_id)
            results.extend(batch_results)
        
        return results
    
    async def _embed_batch(
        self,
        chunks: List[Chunk],
        tenant_id: str
    ) -> List[Dict[str, Any]]:
        texts = [chunk.content for chunk in chunks]
        
        embeddings = await self._get_embeddings(texts, tenant_id)
        
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append({
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
            })
        
        return results
    
    async def _get_embeddings(
        self,
        texts: List[str],
        tenant_id: str
    ) -> List[List[float]]:
        embeddings = []
        uncached_indices = []
        uncached_texts = []
        
        if self._enable_cache and self._cache_port:
            for idx, text in enumerate(texts):
                cache_key = self._get_embedding_cache_key(text)
                
                try:
                    cached = self._cache_port.get(cache_key)
                    if asyncio.iscoroutine(cached):
                        cached = await cached
                    if cached and "embedding" in cached:
                        embeddings.append(cached["embedding"])
                        continue
                except Exception as e:
                    self._logger.warning(f"Cache get failed: {e}")
                
                uncached_indices.append(idx)
                uncached_texts.append(text)
                embeddings.append(None)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts
            embeddings = [None] * len(texts)
        
        if uncached_texts:
            new_embeddings = await self._compute_embeddings(uncached_texts)
            
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
            
            if self._enable_cache and self._cache_port:
                for idx, text in zip(uncached_indices, uncached_texts):
                    cache_key = self._get_embedding_cache_key(text)
                    try:
                        await self._cache_set(
                            cache_key,
                            {"embedding": new_embeddings[uncached_indices.index(idx)]},
                            ttl_seconds=86400 * 30,
                        )
                    except Exception as e:
                        self._logger.warning(f"Cache set failed: {e}")
        
        return embeddings
    
    async def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            if hasattr(self._model, 'aembed'):
                return await self._model.aembed(texts)
            elif hasattr(self._model, 'embed'):
                return self._model.embed(texts)
            else:
                raise RuntimeError("Model has no embed method")
        except Exception as e:
            self._logger.error(f"Embedding failed: {e}")
            raise
    
    def _get_embedding_cache_key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"emb:{self._model_version}:{text_hash}"

    async def _cache_set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        if not self._cache_port:
            return
        result = self._cache_port.set(key, value, ttl_seconds)
        if asyncio.iscoroutine(result):
            await result
