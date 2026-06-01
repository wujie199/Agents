import json
import logging
from typing import Any, Dict, List, Optional


class RagDocumentStore:
    """Structured document table for SQL retrieval backend."""

    def __init__(self, sql_port: Any):
        self._sql = sql_port
        self._logger = logging.getLogger("knowledge.pipeline.index.document_store")
        self._schema_ready = False

    async def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        if hasattr(self._sql, "_init_pool"):
            await self._sql._init_pool()
        self._schema_ready = True

    async def upsert_document(
        self,
        doc_id: str,
        tenant_id: str,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self.ensure_schema()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        title = title or (metadata or {}).get("title") or doc_id
        preview = content[:8000] if content else ""

        existing = await self._sql.select_one(
            "documents",
            ["doc_id"],
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        row = {
            "doc_id": doc_id,
            "tenant_id": tenant_id,
            "title": title,
            "content": preview,
            "metadata": meta_json,
        }
        if existing:
            await self._sql.update(
                "documents",
                {"title": title, "content": preview, "metadata": meta_json},
                {"doc_id": doc_id, "tenant_id": tenant_id},
            )
        else:
            await self._sql.insert("documents", row)

    async def delete_document(self, doc_id: str, tenant_id: str) -> int:
        await self.ensure_schema()
        if hasattr(self._sql, "execute"):
            result = await self._sql.execute(
                "DELETE FROM documents WHERE doc_id = :doc_id AND tenant_id = :tenant_id",
                {"doc_id": doc_id, "tenant_id": tenant_id},
            )
            return len(result) if isinstance(result, list) else 1
        return 0

    async def get_by_doc_id(self, doc_id: str, tenant_id: str) -> Optional[dict]:
        await self.ensure_schema()
        return await self._sql.select_one(
            "documents",
            ["doc_id", "title", "content", "metadata"],
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )

    async def search_by_keywords(
        self,
        tenant_id: str,
        keywords: List[str],
        limit: int = 10,
    ) -> List[dict]:
        await self.ensure_schema()
        if not keywords:
            return []
        clauses = []
        params: Dict[str, Any] = {"tenant_id": tenant_id, "limit": limit}
        for i, kw in enumerate(keywords[:5]):
            key = f"kw{i}"
            clauses.append(f"(title LIKE :{key} OR content LIKE :{key})")
            params[key] = f"%{kw}%"
        where = f"tenant_id = :tenant_id AND ({' OR '.join(clauses)})"
        query = (
            f"SELECT doc_id, title, content, metadata FROM documents "
            f"WHERE {where} LIMIT :limit"
        )
        if hasattr(self._sql, "execute"):
            rows = await self._sql.execute(query, params)
            return [dict(r) for r in rows] if rows else []
        return []
