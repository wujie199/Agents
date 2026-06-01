import asyncio

import pytest

from core.ports.index import IndexProfile, IndexResult
from core.ports.ingest import IngestStatus
from document.rag.bridges.knowledge_base_adapter import KnowledgeBasePortAdapter
from document.rag.config import RagPipelineConfig
from document.rag.pipeline.index.mock_embedding import MockEmbeddingModel
from document.rag.pipeline.index.service import IndexService
from document.rag.pipeline.ingest.adapters.plain_text_adapter import PlainTextIngestAdapter
from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter


class TestIndexPort:
    def test_index_profile_vector_only_skips_side_indexes(self, tmp_path):
        config = RagPipelineConfig(collection_name="idx_prof", enable_graph_index=True)
        config.retrieval.enable_sql = True
        vector = ChromaVectorAdapter(persist_directory=str(tmp_path / "chroma"))
        index = IndexService(
            vector_port=vector,
            embedding_model=MockEmbeddingModel(32),
            config=config,
            sql_port=object(),
            graph_port=object(),
        )
        result = asyncio.run(
            index.index_document(
                "d1",
                "content with DOC-2024 entity",
                "t1",
                profile=IndexProfile.VECTOR_ONLY,
            )
        )
        assert isinstance(result, IndexResult)
        assert result.profile == IndexProfile.VECTOR_ONLY
        assert result.side_indexes.get("sql") is False
        assert result.side_indexes.get("graph") is False


class TestKnowledgeBasePort:
    def test_ingest_and_index(self, tmp_path):
        sample = tmp_path / "kb.txt"
        sample.write_text("ç¥è¯åºé¨é¢æµè¯ï¼ç´¢å¼ä¸æ£ç´¢ã", encoding="utf-8")

        config = RagPipelineConfig(collection_name="kb_test")
        vector = ChromaVectorAdapter(persist_directory=str(tmp_path / "chroma"))
        index = IndexService(
            vector_port=vector,
            embedding_model=MockEmbeddingModel(32),
            config=config,
        )
        kb = KnowledgeBasePortAdapter(
            ingest_port=PlainTextIngestAdapter(),
            index_port=index,
            default_index_profile=IndexProfile.VECTOR_ONLY,
        )
        out = asyncio.run(
            kb.ingest_and_index(
                str(sample),
                doc_id="kb1",
                tenant_id="t1",
            )
        )
        assert out.success is True
        assert out.ingest is not None
        assert out.ingest.status == IngestStatus.SUCCESS
        assert out.index is not None
        assert out.index.chunk_count >= 1

    def test_reindex_deletes_then_writes(self, tmp_path):
        sample = tmp_path / "reindex.txt"
        sample.write_text("ç¬¬ä¸çåå®", encoding="utf-8")

        config = RagPipelineConfig(collection_name="reindex_test")
        vector = ChromaVectorAdapter(persist_directory=str(tmp_path / "chroma2"))
        index = IndexService(
            vector_port=vector,
            embedding_model=MockEmbeddingModel(32),
            config=config,
        )
        kb = KnowledgeBasePortAdapter(
            ingest_port=PlainTextIngestAdapter(),
            index_port=index,
            default_index_profile=IndexProfile.VECTOR_ONLY,
        )
        asyncio.run(kb.ingest_and_index(str(sample), "r1", "t1"))
        sample.write_text("ç¬¬äºçåå®¹æ´é¿ç¨äºéæ°ç´¢å¼æµè¯ã", encoding="utf-8")
        out = asyncio.run(kb.reindex_document(str(sample), "r1", "t1"))
        assert out.success is True
        assert out.index.chunk_count >= 1
