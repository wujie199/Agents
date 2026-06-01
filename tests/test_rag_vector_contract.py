import pytest

from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter
from core.ports.storage.vector import VectorRecord


@pytest.fixture
def vector_store(tmp_path):
    return ChromaVectorAdapter(persist_directory=str(tmp_path / "chroma"))


class TestChromaVectorContract:
    def test_upsert_search_delete(self, vector_store):
        collection = "test_agent"
        records = [
            VectorRecord(
                id="c1",
                vector=[1.0, 0.0, 0.0],
                metadata={"tenant_id": "t1", "doc_id": "d1"},
                content="hello world",
            ),
            VectorRecord(
                id="c2",
                vector=[0.9, 0.1, 0.0],
                metadata={"tenant_id": "t1", "doc_id": "d1"},
                content="hello again",
            ),
        ]
        n = vector_store.upsert(collection, records)
        assert n == 2

        results = vector_store.similarity_search(
            collection,
            [1.0, 0.0, 0.0],
            top_k=2,
            filter={"tenant_id": "t1"},
        )
        assert len(results) >= 1
        assert results[0].id == "c1"

        deleted = vector_store.delete_by_filter(
            collection, {"doc_id": "d1", "tenant_id": "t1"}
        )
        assert deleted == 2
        assert vector_store.count(collection) == 0
