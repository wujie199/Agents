import asyncio

import pytest

from core.domain.context import RequestContext
from core.domain.evidence import SourceType
from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter
from document.rag.bridges.rag_port_adapter import RAGPortAdapter
from document.rag.config import RagPipelineConfig
from document.rag.pipeline.index.mock_embedding import MockEmbeddingModel
from document.rag.query.router.router import RetrievalRouter


class TestRouterDelegation:
    def test_plan_override_vector(self, tmp_path):
        config = RagPipelineConfig(
            collection_name="router_test",
            enable_cache=False,
        )
        config.retrieval.enable_router = True
        config.retrieval.auto_route = False

        vector = ChromaVectorAdapter(persist_directory=str(tmp_path / "chroma"))
        embed = MockEmbeddingModel(48)
        router = RetrievalRouter(
            vector_port=vector,
            embedding_model=embed,
            collection_name=config.collection_name,
            enable_cache=False,
        )
        from document.rag.pipeline.index.service import IndexService

        index = IndexService(vector_port=vector, embedding_model=embed, config=config)
        asyncio.run(
            index.index_document(
                "d1",
                "æ¡¥æ¢æ£æµè§èä¸å»æ¤è¦æ±æè¦ã",
                "t1",
            )
        )

        rag = RAGPortAdapter(
            vector_port=vector,
            embedding_model=embed,
            config=config,
            router=router,
            enable_cache=False,
        )
        request = RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr1",
            channel="test",
        )
        bundle = asyncio.run(
            rag.route_and_retrieve(
                "æ¡¥æ¢æ£æµ",
                request,
                plan={"primary": "vector", "top_k": 5},
            )
        )
        assert len(bundle.evidences) >= 1
        assert bundle.plan.get("primary") == "vector"
        assert bundle.evidences[0].source_type == SourceType.VECTOR

    def test_auto_route_classifies_semantic(self, tmp_path):
        config = RagPipelineConfig(collection_name="rt2", enable_cache=False)
        config.retrieval.enable_router = True
        config.retrieval.auto_route = True

        vector = ChromaVectorAdapter(persist_directory=str(tmp_path / "chroma2"))
        embed = MockEmbeddingModel(32)
        router = RetrievalRouter(
            vector_port=vector,
            embedding_model=embed,
            collection_name="rt2",
            enable_cache=False,
        )
        rag = RAGPortAdapter(
            vector_port=vector,
            embedding_model=embed,
            config=config,
            router=router,
            enable_cache=False,
        )
        request = RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr2",
            channel="test",
        )
        from document.rag.pipeline.index.service import IndexService

        index = IndexService(vector_port=vector, embedding_model=embed, config=config)
        asyncio.run(
            index.index_document("d1", "å¦ä½è¿è¡ä»£ç å®¡æ¥çæä½³å®è·µã", "t1")
        )
        bundle = asyncio.run(
            rag.route_and_retrieve("å¦ä½è¿è¡ä»£ç å®¡æ¥ï¼", request)
        )
        assert "primary" in (bundle.plan or {})
