import asyncio
import logging
from typing import Any, List, Optional

from core.ports.storage.vector import VectorRecord


class SessionMessageVectorIndex:
    """L2 可选向量索引：写入时 embed，检索时 similarity_search。"""

    def __init__(
        self,
        vector_port: Any,
        embedding_model: Any,
        collection: str = "session_messages",
        *,
        embed_batch_size: int = 32,
        index_version: str = "v1",
    ):
        self._vector = vector_port
        self._embedding = embedding_model
        self._collection = collection
        self._embed_batch_size = max(1, embed_batch_size)
        self._index_version = index_version
        self._logger = logging.getLogger(__name__)

    @property
    def index_version(self) -> str:
        return self._index_version

    def get_stored_version(self) -> str:
        getter = getattr(self._vector, "get_index_version", None)
        if getter is None:
            return ""
        return str(getter(self._collection))

    def is_version_current(self) -> bool:
        stored = self.get_stored_version()
        if not stored or stored == "v1":
            count = getattr(self._vector, "count", None)
            if count is not None and count(self._collection) == 0:
                return True
        return stored == self._index_version

    def mark_index_current(self) -> None:
        setter = getattr(self._vector, "set_index_version", None)
        if setter is None:
            return
        setter(self._collection, self._index_version)

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self._embedding, "aembed"):
            return await self._embedding.aembed(texts)
        if hasattr(self._embedding, "embed"):
            return await asyncio.to_thread(self._embedding.embed, texts)
        raise RuntimeError("embedding model has no embed/aembed")

    async def _upsert_text(
        self,
        *,
        record_id: str,
        content: str,
        metadata: dict,
    ) -> None:
        vectors = await self._embed([content])
        vector_record = VectorRecord(
            id=record_id,
            vector=vectors[0],
            metadata=metadata,
            content=content,
        )
        await asyncio.to_thread(
            self._vector.upsert,
            self._collection,
            [vector_record],
        )

    async def index_message(self, record: dict) -> None:
        content = (record.get("content") or "").strip()
        if not content or content == "[redacted]" or record.get("redacted"):
            return

        message_id = record["message_id"]
        metadata = {
            "tenant_id": str(record.get("tenant_id", "")),
            "user_id": str(record.get("user_id", "")),
            "session_id": str(record.get("session_id", "")),
            "message_id": message_id,
            "role": str(record.get("role", "user")),
            "ts": str(record.get("ts", "")),
            "record_type": "message",
        }
        await self._upsert_text(
            record_id=message_id,
            content=content,
            metadata=metadata,
        )

    async def index_tool_call(self, record: dict) -> None:
        tool_name = (record.get("tool_name") or "").strip()
        summary = (record.get("result_summary") or "").strip()
        if summary == "[redacted]":
            return
        content = f"{tool_name}: {summary}".strip(": ")
        if not content:
            return

        call_id = record["call_id"]
        metadata = {
            "tenant_id": str(record.get("tenant_id", "")),
            "user_id": str(record.get("user_id", "")),
            "session_id": str(record.get("session_id", "")),
            "message_id": call_id,
            "role": "tool",
            "ts": str(record.get("ts", "")),
            "record_type": "tool_call",
        }
        await self._upsert_text(
            record_id=call_id,
            content=content,
            metadata=metadata,
        )

    def _build_filter(
        self,
        tenant_id: str,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> dict:
        filt = {"tenant_id": tenant_id, "user_id": user_id}
        if session_id:
            filt["session_id"] = session_id
        return filt

    async def search(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        *,
        session_id: Optional[str] = None,
        limit: int = 15,
    ) -> List[dict]:
        vectors = await self._embed([query])
        results = await asyncio.to_thread(
            self._vector.similarity_search,
            self._collection,
            vectors[0],
            limit,
            self._build_filter(tenant_id, user_id, session_id),
        )
        messages: List[dict] = []
        for hit in results:
            meta = hit.metadata or {}
            content = hit.content or ""
            if not content or content == "[redacted]":
                continue
            role = meta.get("role", "user")
            messages.append(
                {
                    "message_id": meta.get("message_id") or hit.id,
                    "session_id": meta.get("session_id", ""),
                    "role": role,
                    "content": content,
                    "ts": meta.get("ts", ""),
                    "redacted": 0,
                    "vector_score": hit.score,
                    "source": "vector",
                }
            )
        return messages

    async def delete_message(self, message_id: str) -> None:
        await asyncio.to_thread(
            self._vector.delete_by_ids,
            self._collection,
            [message_id],
        )

    async def delete_user_messages(self, tenant_id: str, user_id: str) -> int:
        deleter = getattr(self._vector, "delete_by_filter", None)
        if deleter is None:
            return 0
        return await asyncio.to_thread(
            deleter,
            self._collection,
            {"tenant_id": tenant_id, "user_id": user_id},
        )

    async def delete_session_messages(self, session_id: str) -> int:
        deleter = getattr(self._vector, "delete_by_filter", None)
        if deleter is None:
            return 0
        return await asyncio.to_thread(
            deleter,
            self._collection,
            {"session_id": session_id},
        )
