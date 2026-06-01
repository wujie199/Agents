import asyncio
import pytest

from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter
from document.rag.pipeline.index.service import IndexService
from document.rag.pipeline.index.mock_embedding import MockEmbeddingModel
from document.rag.config import RagPipelineConfig


@pytest.fixture
def index_service(tmp_path):
    config = RagPipelineConfig(
        collection_name="test_index",
        chunk_size=100,
        chunk_overlap=10,
    )
    vector = ChromaVectorAdapter(persist_directory=str(tmp_path / "chroma"))
    return IndexService(
        vector_port=vector,
        embedding_model=MockEmbeddingModel(dimension=32),
        config=config,
    )


class TestIndexService:
    def test_index_and_search(self, index_service):
        result = asyncio.run(index_service.index_document(
            doc_id="doc1",
            content="RAG pipeline test content for indexing and search.",
            tenant_id="t1",
        ))
        assert result.chunk_count >= 1
        assert result.vectors_written >= 1

        vector = index_service._vector_port
        query_vec = MockEmbeddingModel(dimension=32).embed(["indexing search"])[0]
        hits = vector.similarity_search(
            "test_index",
            query_vec,
            top_k=3,
            filter={"tenant_id": "t1"},
        )
        assert len(hits) >= 1

    def test_delete_document(self, index_service):
        asyncio.run(index_service.index_document(
            doc_id="doc2",
            content="temporary document to delete",
            tenant_id="t2",
        ))
        ok = asyncio.run(index_service.delete_document("doc2", "t2"))
        assert ok is True
