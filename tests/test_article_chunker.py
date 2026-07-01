"""Article chunker tests for contract documents."""

from document.rag.application.indexing.article_chunker import ArticleChunker

SAMPLE = """
第一章 总则

第一条 本合同由甲乙双方签署。
第二条 双方应遵守相关法律法规。

第二章 权利义务

第三条 甲方应按时付款。
第四条 乙方应按时交付。
"""


def test_article_chunker_splits_by_article():
    chunker = ArticleChunker(chunk_size=2000, chunk_overlap=0)
    chunks = chunker.chunk(SAMPLE, "doc1")
    assert len(chunks) >= 4
    numbers = {c.metadata.get("header", "") for c in chunks}
    assert any("第一条" in h for h in numbers)
    assert any("第四条" in h for h in numbers)


def test_article_chunker_section_path():
    chunker = ArticleChunker(chunk_size=2000, chunk_overlap=0)
    chunks = chunker.chunk(SAMPLE, "doc1")
    third = next(c for c in chunks if "第三条" in c.metadata.get("header", ""))
    path = third.metadata.get("section_path", "")
    assert "第二章" in path
    assert "第三条" in path


def test_article_chunker_metadata_strategy():
    chunker = ArticleChunker()
    chunks = chunker.chunk("第一条 测试内容。", "d")
    assert chunks[0].metadata.get("strategy") == "article"
