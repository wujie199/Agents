"""KnowledgeBasePortAdapter 格式清理接入测试。"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.ports.cleaner import CleaningLevel
from core.ports.index import IndexProfile, IndexResult
from core.ports.ingest import DocumentFormat, IngestResult, IngestStatus
from document.rag.facades.knowledge_base import KnowledgeBasePortAdapter


@pytest.mark.asyncio
async def test_ingest_and_index_applies_cleaning(tmp_path):
    html_path = tmp_path / "page.html"
    html_path.write_text("<p>Hello</p>   <b>World</b>", encoding="utf-8")

    ingest_result = IngestResult(
        content="<p>Hello</p>   <b>World</b>",
        metadata={"ingest_backend": "plain_text", "doc_id": "h1"},
        status=IngestStatus.SUCCESS,
        pages=[{"page_num": 1, "content": "<p>Hello</p>   <b>World</b>"}],
    )

    mock_ingest = MagicMock()
    mock_ingest.ingest_from_path.return_value = ingest_result

    mock_index = MagicMock()
    mock_index.index_from_ingest = AsyncMock(
        return_value=IndexResult(
            doc_id="h1",
            chunk_count=1,
            vectors_written=1,
            collection="agent",
            indexed_at="2026-01-01T00:00:00",
            model_version="v1",
            profile=IndexProfile.VECTOR_ONLY,
        )
    )

    kb = KnowledgeBasePortAdapter(
        ingest_port=mock_ingest,
        index_port=mock_index,
        enable_cleaning=True,
        ocr_postprocess=True,
        cleaning_level=CleaningLevel.STANDARD,
    )

    result = await kb.ingest_and_index(
        file_path=str(html_path),
        doc_id="h1",
        tenant_id="t1",
        index_profile=IndexProfile.VECTOR_ONLY,
    )

    assert result.success is True
    assert result.ingest.metadata.get("cleaned") is True
    assert "<p>" not in result.ingest.content
    assert "Hello" in result.ingest.content


@pytest.mark.asyncio
async def test_ingest_and_index_skips_cleaning_when_disabled(tmp_path):
    txt_path = tmp_path / "note.txt"
    txt_path.write_text("raw  text", encoding="utf-8")

    ingest_result = IngestResult(
        content="raw  text",
        metadata={"ingest_backend": "plain_text"},
        status=IngestStatus.SUCCESS,
    )

    mock_ingest = MagicMock()
    mock_ingest.ingest_from_path.return_value = ingest_result

    mock_index = MagicMock()
    mock_index.index_from_ingest = AsyncMock(
        return_value=IndexResult(
            doc_id="n1",
            chunk_count=1,
            vectors_written=1,
            collection="agent",
            indexed_at="2026-01-01T00:00:00",
            model_version="v1",
            profile=IndexProfile.VECTOR_ONLY,
        )
    )

    kb = KnowledgeBasePortAdapter(
        ingest_port=mock_ingest,
        index_port=mock_index,
        enable_cleaning=False,
    )

    result = await kb.ingest_and_index(
        file_path=str(txt_path),
        doc_id="n1",
        tenant_id="t1",
        index_profile=IndexProfile.VECTOR_ONLY,
    )

    assert result.success is True
    assert result.ingest.metadata.get("cleaned") is None
    assert result.ingest.content == "raw  text"
