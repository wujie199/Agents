"""七步分块流水线测试。"""

from document.rag.config.chunk_pipeline import ChunkPipelineConfig, parse_chunk_pipeline_config
from document.rag.application.chunking.pipeline import SevenStepChunkPipeline
from document.rag.application.chunking.chunker import SevenStepChunker
from core.ports.chunker import ChunkStrategy
from document.rag.application.indexing.chunker import create_chunker, parse_chunk_strategy


SAMPLE_MD = """# 安装指南

## 依赖检查

Python 3.11 是指本系统推荐的运行环境。请先安装依赖包。

另外还需要配置数据库连接。例如：使用 SQLite 作为默认存储。

## 启动步骤

运行 `python app.py` 启动服务。该系统会自动加载配置。

| 参数 | 说明 |
|------|------|
| port | 端口号 |
| host | 绑定地址 |
"""


SAMPLE_FAQ = """1、什么是扫地机器人？
扫地机器人是一种自动清洁设备。

2、如何保养滤网？
定期用清水冲洗滤网并晾干。
"""


class TestChunkPipelineConfig:
    def test_parse_from_dict(self):
        cfg = parse_chunk_pipeline_config({"domain": "legal", "target_ideal": 200})
        assert cfg.domain == "legal"
        assert cfg.target_min == 100

    def test_with_domain_faq(self):
        cfg = ChunkPipelineConfig(domain="faq").with_domain("faq")
        assert cfg.target_max == 0


class TestSevenStepPipeline:
    def test_markdown_produces_retrieval_chunks(self):
        cfg = ChunkPipelineConfig(domain="general", enable_parent_child=True)
        pipeline = SevenStepChunkPipeline(cfg)
        result = pipeline.run(SAMPLE_MD, "doc_test", {"format": "md"})
        assert result.stats["units"] >= 3
        assert len(result.retrieval_chunks) >= 1
        assert all(ch.content.strip() for ch in result.retrieval_chunks)

    def test_table_unit_preserved(self):
        cfg = ChunkPipelineConfig(domain="general")
        pipeline = SevenStepChunkPipeline(cfg)
        result = pipeline.run(SAMPLE_MD, "doc_table", {"format": "md"})
        table_chunks = [c for c in result.retrieval_chunks if c.unit_type == "table"]
        assert table_chunks or any("port" in c.content for c in result.retrieval_chunks)

    def test_faq_whole_pair(self):
        cfg = ChunkPipelineConfig(domain="faq", preserve_faq_pairs=True)
        pipeline = SevenStepChunkPipeline(cfg)
        result = pipeline.run(SAMPLE_FAQ, "doc_faq", {"chunk_domain": "faq"})
        assert len(result.retrieval_chunks) >= 2

    def test_all_seven_steps_run(self):
        cfg = ChunkPipelineConfig(domain="general")
        pipeline = SevenStepChunkPipeline(cfg)
        result = pipeline.run(SAMPLE_MD, "doc_stats", {"format": "md"})
        stats = result.stats
        assert "units" in stats
        assert "confirmed_cuts" in stats
        assert "base_chunks" in stats
        assert "retrieval_chunks" in stats
        assert "repair_tasks" in stats


class TestSevenStepChunker:
    def test_create_chunker_strategy(self):
        assert parse_chunk_strategy("seven_step") == ChunkStrategy.SEVEN_STEP
        chunker = create_chunker(
            ChunkStrategy.SEVEN_STEP,
            pipeline_cfg=ChunkPipelineConfig(domain="general"),
        )
        assert isinstance(chunker, SevenStepChunker)

    def test_chunker_returns_chunks(self):
        chunker = SevenStepChunker(pipeline_cfg=ChunkPipelineConfig(domain="general"))
        chunks = chunker.chunk(SAMPLE_MD, "doc1", {"format": "md", "strategy": "seven_step"})
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("strategy") == "seven_step"
        assert "heading_path" in chunks[0].metadata
