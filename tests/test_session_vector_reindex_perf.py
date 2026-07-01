"""Session 向量 reindex 批量与跳过。"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_platform.memory.adapters.session_message_vector_index import (
    SessionMessageVectorIndex,
)


class _BatchEmbedding:
    def __init__(self):
        self.calls: list[list[str]] = []

    async def aembed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t))] for t in texts]


class _VectorPort:
    def __init__(self):
        self.upserts: list[int] = []

    def upsert(self, collection, records):
        self.upserts.append(len(records))
        return len(records)

    def count_by_filter(self, collection, filt):
        return 2 if filt.get("session_id") == "s1" else 0

    def get_index_version(self, collection):
        return "v-test"

    def set_index_version(self, collection, version):
        pass


@pytest.mark.asyncio
async def test_index_messages_batches_embed():
    emb = _BatchEmbedding()
    vec = _VectorPort()
    idx = SessionMessageVectorIndex(vec, emb, embed_batch_size=32)
    records = [
        {
            "message_id": f"m{i}",
            "content": f"msg{i}",
            "session_id": "s1",
            "role": "user",
        }
        for i in range(40)
    ]
    n = await idx.index_messages(records)
    assert n == 40
    assert len(emb.calls) == 2
    assert emb.calls[0] == [f"msg{i}" for i in range(32)]
    assert vec.upserts == [32, 8]


@pytest.mark.asyncio
async def test_reindex_skips_when_vectors_cover_archive():
    from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter

    memory = MemoryPortAdapter.__new__(MemoryPortAdapter)
    memory._archive_db = MagicMock()
    memory._logger = MagicMock()

    vector_index = MagicMock()
    vector_index.is_version_current.return_value = True
    vector_index.index_version = "v-test"
    vector_index.count_session_vectors.return_value = 5
    memory._vector_index = vector_index

    memory._count_indexable_messages = AsyncMock(return_value=5)

    result = await MemoryPortAdapter.reindex_session_vectors(
        memory,
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
    )
    assert result.get("skipped") is True
    assert result.get("reason") == "already_indexed"
