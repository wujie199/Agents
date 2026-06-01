import asyncio

import pytest

from core.domain.context import RequestContext
from core.domain.evidence import Evidence, SourceType
from document.rag.bridges.rag_port_adapter import RAGPortAdapter
from document.rag.config import RagPipelineConfig
from document.rag.pipeline.index.mock_embedding import MockEmbeddingModel
from document.rag.pipeline.index.mock_rerank import MockRerankModel
from document.rag.pipeline.index.service import IndexService
from document.rag.query.retrieval.rerank_utils import apply_rerank
from document.rag.query.rewrite.multi_query import QueryRewriterPipeline
from document.rag.query.router.router import RetrievalRouter
from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter
from agent_platform.storage.adapters.graph.memory_graph_adapter import MemoryGraphAdapter
from agent_platform.storage.adapters.sqlite.relational_adapter import AsyncSQLiteRelationalAdapter


class _MultiQueryMock:
    async def expand(self, query: str):
        return [query, f"{query} æ©å±"]


class TestTriBackendRerankRewrite:
    @pytest.fixture
    def stack(self, tmp_path):
        config = RagPipelineConfig(
            collection_name="tri",
            enable_cache=False,
            enable_graph_index=True,
        )
        config.retrieval.enable_router = True
        config.retrieval.enable_sql = True
        config.retrieval.enable_graph = True
        config.retrieval.enable_rerank = True
        config.rewrite.enable_multi_query = True

        vector = ChromaVectorAdapter(persist_directory=str(tmp_path / "chroma"))
        sql = AsyncSQLiteRelationalAdapter(db_path=str(tmp_path / "rag.db"))
        graph = MemoryGraphAdapter()
        embed = MockEmbeddingModel(48)
        rerank = MockRerankModel()
        rewriter = QueryRewriterPipeline(
            multi_query_expander=_MultiQueryMock(),
            enable_multi_query=True,
        )

        router = RetrievalRouter(
            vector_port=vector,
            embedding_model=embed,
            sql_port=sql,
            graph_port=graph,
            rerank_model=rerank,
            query_rewriter=rewriter,
            collection_name="tri",
            enable_cache=False,
            enable_sql=True,
            enable_graph=True,
            enable_rerank=True,
        )
        index = IndexService(
            vector_port=vector,
            embedding_model=embed,
            config=config,
            sql_port=sql,
            graph_port=graph,
        )
        rag = RAGPortAdapter(
            vector_port=vector,
            embedding_model=embed,
            config=config,
            rerank_model=rerank,
            router=router,
            enable_cache=False,
        )
        return rag, index, graph

    def test_sql_retrieval_by_entity(self, stack):
        rag, index, _ = stack
        asyncio.run(
            index.index_document(
                "DOC-2024",
                "é¡¹ç® DOC-2024 çæ¡¥æ¢æ£æµè§èä¸å»æ¤è¦æ±ã",
                "t1",
                metadata={"title": "æ¡¥æ¢è§è"},
            )
        )
        ctx = RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr1",
            channel="test",
        )
        bundle = asyncio.run(
            rag.route_and_retrieve(
                "æ¥æ¾ DOC-2024",
                ctx,
                plan={"primary": "sql", "top_k": 5, "rerank_top_n": 3},
            )
        )
        assert not bundle.empty
        assert any(e.source_type == SourceType.SQL for e in bundle.evidences)

    def test_graph_retrieval(self, stack):
        rag, index, graph = stack
        asyncio.run(
            index.index_document(
                "PRJ-99",
                "è´è´£äººå³èå®ä½?REQ-100 ä¸?DOC-2024ã",
                "t1",
            )
        )
        ctx = RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr2",
            channel="test",
        )
        bundle = asyncio.run(
            rag.route_and_retrieve(
                "è°è´è´?PRJ-99ï¼",
                ctx,
                plan={
                    "primary": "graph",
                    "secondary": ["vector"],
                    "graph_hop": 2,
                    "top_k": 5,
                },
            )
        )
        assert not bundle.empty
        types = {e.source_type for e in bundle.evidences}
        assert SourceType.GRAPH in types or SourceType.VECTOR in types
        assert graph.health()["nodes"] >= 1

    def test_multi_query_rewrite_expands(self, stack):
        rag, index, _ = stack
        asyncio.run(
            index.index_document("d1", "ä»£ç å®¡æ¥æä½³å®è·µä¸æµç¨è¯´æã", "t1")
        )
        ctx = RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr3",
            channel="test",
        )
        bundle = asyncio.run(
            rag.route_and_retrieve(
                "å¦ä½è¿è¡ä»£ç å®¡æ¥ï¼",
                ctx,
                plan={"primary": "vector", "top_k": 5},
            )
        )
        assert not bundle.empty

    def test_rerank_utils(self):
        ev = Evidence(
            id="1",
            content="æ¡¥æ¢æ£æµè§è",
            score=0.5,
            source_type=SourceType.VECTOR,
        )
        out = asyncio.run(
            apply_rerank(MockRerankModel(), "æ¡¥æ¢æ£æµ", [ev], 1)
        )
        assert len(out) == 1
        assert out[0].metadata.get("rerank_score") is not None
