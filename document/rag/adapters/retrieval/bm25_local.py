"""本地 BM25 倒排索引（建库写入、查询检索）。"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_log = logging.getLogger("document.rag.adapters.retrieval.bm25_local")

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_PATTERN.findall(text) if t.strip()]


@dataclass
class Bm25Document:
    doc_id: str
    chunk_id: str
    content: str
    tenant_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: List[str] = field(default_factory=list)


class Bm25Okapi:
    """轻量 Okapi BM25（无第三方依赖）。"""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._corpus = [list(doc) for doc in corpus]
        self._nd = len(self._corpus)
        self._avgdl = sum(len(d) for d in self._corpus) / self._nd if self._nd else 0.0
        df: Counter[str] = Counter()
        for doc in self._corpus:
            df.update(set(doc))
        self._idf = {
            term: math.log(1 + (self._nd - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        if not self._corpus:
            return []
        scores = [0.0] * self._nd
        q_terms = Counter(query_tokens)
        for term, qf in q_terms.items():
            if term not in self._idf:
                continue
            idf = self._idf[term]
            for i, doc in enumerate(self._corpus):
                tf = doc.count(term)
                if tf == 0:
                    continue
                dl = len(doc)
                denom = tf + self._k1 * (1 - self._b + self._b * dl / (self._avgdl or 1))
                scores[i] += idf * tf * (self._k1 + 1) / denom * qf
        return scores


class LocalBm25Index:
    """
    按 collection 持久化的 BM25 索引。
    路径: {data_dir}/bm25_index/{collection}.json
    """

    def __init__(self, index_path: Path):
        self._path = index_path
        self._documents: List[Bm25Document] = []
        self._model: Optional[Bm25Okapi] = None
        self._load()

    @classmethod
    def for_collection(cls, data_dir: Path, collection: str) -> "LocalBm25Index":
        root = data_dir / "bm25_index"
        root.mkdir(parents=True, exist_ok=True)
        return cls(root / f"{collection}.json")

    def _load(self) -> None:
        if not self._path.is_file():
            self._documents = []
            self._rebuild_model()
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
            self._documents = [
                Bm25Document(
                    doc_id=str(d["doc_id"]),
                    chunk_id=str(d["chunk_id"]),
                    content=str(d.get("content") or ""),
                    tenant_id=str(d.get("tenant_id") or ""),
                    metadata=dict(d.get("metadata") or {}),
                    tokens=list(d.get("tokens") or tokenize(str(d.get("content") or ""))),
                )
                for d in (raw.get("documents") or [])
            ]
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            _log.warning("BM25 索引加载失败，将重建空索引: %s", exc)
            self._documents = []
        self._rebuild_model()

    def _rebuild_model(self) -> None:
        corpus = [d.tokens for d in self._documents]
        self._model = Bm25Okapi(corpus) if corpus else None

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "chunk_id": d.chunk_id,
                    "content": d.content,
                    "tenant_id": d.tenant_id,
                    "metadata": d.metadata,
                    "tokens": d.tokens,
                }
                for d in self._documents
            ],
        }
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(self._path)

    def delete_by_doc_id(self, doc_id: str, tenant_id: Optional[str] = None) -> int:
        before = len(self._documents)
        if tenant_id:
            self._documents = [
                d
                for d in self._documents
                if not (d.doc_id == doc_id and d.tenant_id == tenant_id)
            ]
        else:
            self._documents = [d for d in self._documents if d.doc_id != doc_id]
        removed = before - len(self._documents)
        if removed:
            self._rebuild_model()
            self.save()
        return removed

    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        tenant_id: str,
        doc_id: str,
    ) -> int:
        """替换某 doc 的全部 chunk 条目后写入 BM25。"""
        self.delete_by_doc_id(doc_id, tenant_id)
        added = 0
        for item in chunks:
            content = str(item.get("content") or "")
            chunk_id = str(item.get("chunk_id") or item.get("id") or "")
            if not chunk_id or not content.strip():
                continue
            meta = dict(item.get("metadata") or {})
            meta.setdefault("doc_id", doc_id)
            meta.setdefault("tenant_id", tenant_id)
            self._documents.append(
                Bm25Document(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    content=content,
                    tenant_id=tenant_id,
                    metadata=meta,
                    tokens=tokenize(content),
                )
            )
            added += 1
        self._rebuild_model()
        self.save()
        _log.info("BM25 indexed doc_id=%s chunks=%d total=%d", doc_id, added, len(self._documents))
        return added

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self._model or not self._documents or not query.strip():
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._model.get_scores(q_tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )
        results: List[Dict[str, Any]] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            doc = self._documents[idx]
            if tenant_id and doc.tenant_id != tenant_id:
                continue
            results.append(
                {
                    "id": doc.chunk_id,
                    "content": doc.content,
                    "score": float(score),
                    "metadata": {**doc.metadata, "retrieval_backend": "bm25"},
                }
            )
            if len(results) >= top_k:
                break
        return results

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def rebuild_from_chroma(
        self,
        chroma_dir: str,
        collection: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> int:
        """从已有 Chroma 集合全量重建 BM25（升级或首次启用 hybrid 时使用）。"""
        import chromadb

        client = chromadb.PersistentClient(path=chroma_dir)
        col = client.get_or_create_collection(collection)
        where = {"tenant_id": tenant_id} if tenant_id else None
        if where and len(where) > 1:
            where = {"$and": [{k: v} for k, v in where.items()]}
        batch = col.get(where=where, include=["documents", "metadatas"])
        ids = batch.get("ids") or []
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []
        rebuilt: List[Bm25Document] = []
        for chunk_id, content, meta in zip(ids, documents, metadatas):
            meta = dict(meta or {})
            doc_id = str(meta.get("doc_id") or "")
            tid = str(meta.get("tenant_id") or tenant_id or "")
            text = str(content or "")
            if not chunk_id or not text.strip():
                continue
            rebuilt.append(
                Bm25Document(
                    doc_id=doc_id,
                    chunk_id=str(chunk_id),
                    content=text,
                    tenant_id=tid,
                    metadata=meta,
                    tokens=tokenize(text),
                )
            )
        if tenant_id:
            # 仅替换指定租户，保留其他租户的 BM25 条目
            self._documents = [
                d for d in self._documents if d.tenant_id != tenant_id
            ]
            self._documents.extend(rebuilt)
            added = len(rebuilt)
        else:
            self._documents = rebuilt
            added = len(rebuilt)
        self._rebuild_model()
        self.save()
        _log.info(
            "BM25 rebuilt from Chroma collection=%s tenant=%s added=%d total=%d path=%s",
            collection,
            tenant_id or "*",
            added,
            len(self._documents),
            self._path,
        )
        return added
