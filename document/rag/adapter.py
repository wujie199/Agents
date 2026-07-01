"""Agent 侧极简 RAG 适配器。

Agent 只需调用 retrieve() / index_file() / delete() 即可，
RAG 内部的向量/混合/路由/缓存/重排复杂性完全透明。
"""

from typing import Any, Optional

from core.domain.context import RequestContext
from core.domain.evidence import EvidenceBundle
from core.ports.knowledge_base import IngestAndIndexResult


class RagAdapter:
    """Agent 调用 RAG 的唯一入口。

    用法::

        from document.rag.adapter import RagAdapter
        adapter = RagAdapter(rag_stack)   # 由 bootstrap 构建
        bundle = await adapter.retrieve("用户问题", context)
    """

    def __init__(self, rag_stack: Any):
        """
        Args:
            rag_stack: document.rag.bootstrap.online.RagStack 实例
        """
        self._rag = rag_stack.rag
        self._kb = rag_stack.knowledge_base
        self._index = rag_stack.index_port
        self._config = rag_stack.config

    # ── 检索 ──

    async def retrieve(
        self, query: str, context: RequestContext, *, plan: dict | None = None
    ) -> EvidenceBundle:
        """检索知识证据（向量/混合/路由 — 自动选择）。"""
        return await self._rag.route_and_retrieve(query, context, plan=plan)

    async def retrieve_batch(
        self,
        requests: list,
        context: RequestContext,
        *,
        plan: dict | None = None,
    ) -> list[EvidenceBundle]:
        """批量检索。"""
        return await self._rag.route_and_retrieve_batch(requests, context, plan=plan)

    # ── 文档管理 ──

    async def index_file(
        self, file_path: str, doc_id: str, tenant_id: str, **kw
    ) -> IngestAndIndexResult:
        """摄取并索引文档。"""
        return await self._kb.ingest_and_index(file_path, doc_id, tenant_id, **kw)

    async def delete(self, doc_id: str, tenant_id: str) -> None:
        """删除文档。"""
        return await self._rag.invalidate_document(doc_id, tenant_id)

    # ── 健康检查 ──

    async def health(self) -> dict:
        return await self._rag.health()

    def get_cache_stats(self) -> dict:
        return self._rag.get_cache_stats()

    # ── RAGPort 协议兼容 ──
    # 以下方法确保 RagAdapter 满足 core.ports.rag.RAGPort Protocol

    async def route_and_retrieve(
        self, query: str, context: RequestContext, plan: Optional[Any] = None
    ) -> EvidenceBundle:
        return await self._rag.route_and_retrieve(query, context, plan=plan)

    async def route_and_retrieve_batch(
        self,
        requests: list,
        context: RequestContext,
        plan: Optional[Any] = None,
    ) -> list[EvidenceBundle]:
        return await self._rag.route_and_retrieve_batch(requests, context, plan=plan)

    async def invalidate_document(self, doc_id: str, tenant_id: str) -> None:
        return await self._rag.invalidate_document(doc_id, tenant_id)
