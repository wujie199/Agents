import asyncio
import pytest
from pathlib import Path

from core.domain.context import RequestContext
from core.domain.evidence import SourceType
from document.rag.pipeline.ingest.factory import build_ingest_pipeline
from document.rag.pipeline.index.mock_embedding import MockEmbeddingModel
from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter
from document.rag.bridges.rag_port_adapter import RAGPortAdapter
from document.rag.pipeline.index.service import IndexService
from document.rag.config import RagPipelineConfig
from agent_platform.storage.adapters.memory.async_cache_adapter import AsyncMemoryCacheAdapter


class TestRagE2E:
    def test_ingest_index_retrieve(self, tmp_path):
        sample = tmp_path / "sample.txt"
        sample.write_text(
            "å·¥ç¨æ¥å RAG æµè¯ï¼æ¡¥æ¢æ£æµä¸å»æ¤è§èæè¦ã",
            encoding="utf-8",
        )

        request = RequestContext(
            tenant_id="t_e2e",
            user_id="u1",
            session_id="s1",
            trace_id="tr1",
            channel="test",
        )

        config = RagPipelineConfig(collection_name="e2e_agent", enable_cache=False)
        vector = ChromaVectorAdapter(persist_directory=str(tmp_path / "chroma"))
        cache = AsyncMemoryCacheAdapter()
        embed = MockEmbeddingModel(dimension=48)

        index_service = IndexService(
            vector_port=vector,
            embedding_model=embed,
            config=config,
            cache_port=cache,
        )
        rag = RAGPortAdapter(
            vector_port=vector,
            cache_port=cache,
            embedding_model=embed,
            config=config,
            enable_cache=False,
        )

        ingest = build_ingest_pipeline(config)
        ingest_result = ingest.ingest_from_path(str(sample), doc_id="sample_doc")
        assert ingest_result.status.value in ("success", "partial")

        asyncio.run(index_service.index_from_ingest(ingest_result, tenant_id="t_e2e"))

        bundle = asyncio.run(rag.route_and_retrieve("æ¡¥æ¢æ£æµ", request))
        assert len(bundle.evidences) >= 1
        assert bundle.evidences[0].source_type == SourceType.VECTOR
        assert "æ¡¥æ¢" in bundle.evidences[0].content or bundle.evidences[0].score > 0

    def test_empty_index_returns_degraded(self, tmp_path):
        request = RequestContext(
            tenant_id="t_empty",
            user_id="u1",
            session_id="s1",
            trace_id="tr2",
            channel="test",
        )
        config = RagPipelineConfig(collection_name="empty_agent")
        vector = ChromaVectorAdapter(persist_directory=str(tmp_path / "chroma2"))
        embed = MockEmbeddingModel()
        rag = RAGPortAdapter(
            vector_port=vector,
            embedding_model=embed,
            config=config,
            enable_cache=False,
        )
        bundle = asyncio.run(rag.route_and_retrieve("anything", request))
        assert bundle.empty is True
        assert bundle.is_degraded() is True
