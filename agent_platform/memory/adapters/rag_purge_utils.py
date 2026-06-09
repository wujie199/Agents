"""purge_user 时联动 RAG 文档删除。"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


async def list_rag_document_ids_for_user(
    archive_db: Any,
    tenant_id: str,
    user_id: str,
) -> List[str]:
    """从 documents.metadata 中匹配 user_id / owner_id。"""
    selector = getattr(archive_db, "select_many", None)
    if selector is None:
        return []
    try:
        rows = await selector(
            "documents",
            ["doc_id", "metadata"],
            where={"tenant_id": tenant_id},
            limit=10_000,
        )
    except Exception as e:
        logger.warning("List RAG documents failed: %s", e)
        return []

    doc_ids: List[str] = []
    for row in rows:
        meta_raw = row.get("metadata") or "{}"
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
        except json.JSONDecodeError:
            meta = {}
        owner = meta.get("user_id") or meta.get("owner_id") or meta.get("uploaded_by")
        if owner == user_id:
            doc_ids.append(str(row["doc_id"]))
    return doc_ids


async def purge_rag_documents_for_user(
    index_port: Any,
    archive_db: Any,
    tenant_id: str,
    user_id: str,
) -> dict:
    if index_port is None:
        return {"deleted": 0, "doc_ids": [], "reason": "index_port_not_configured"}

    doc_ids = await list_rag_document_ids_for_user(archive_db, tenant_id, user_id)
    deleted = 0
    errors = 0
    for doc_id in doc_ids:
        try:
            ok = await index_port.delete_document(doc_id, tenant_id)
            if ok:
                deleted += 1
        except Exception as e:
            errors += 1
            logger.warning("RAG delete doc_id=%s failed: %s", doc_id, e)
    return {"deleted": deleted, "errors": errors, "doc_ids": doc_ids}
