import pytest
from document.rag.query.router.classifier import QueryClassifier, QueryType, ClassificationResult
from document.rag.query.router.rules import RoutingRules, RetrievalPlan, BackendType
from document.rag.query.router.fusion import RRFFusion, WeightedFusion, CascadeFusion, FusionFactory
from document.rag.pipeline.index.chunker import RecursiveChunker, MarkdownChunker, create_chunker
from document.rag.pipeline.index.embedder import Embedder
from document.rag.query.rewrite.hyde import HyDERewriter
from document.rag.query.rewrite.multi_query import MultiQueryExpander, QueryRewriterPipeline
from core.ports.chunker import ChunkStrategy
from core.domain.evidence import Evidence


class TestQueryClassifier:
    
    @pytest.fixture
    def classifier(self):
        return QueryClassifier()
    
    def test_classify_exact_id(self, classifier):
        result = classifier.classify("PRJ-12345")
        
        assert result.query_type == QueryType.FACTUAL_EXACT
        assert result.confidence > 0.9
    
    def test_classify_graph_query(self, classifier):
        result = classifier.classify("è°è´è´£è¿ä¸ªé¡¹ç®ï¼")
        
        assert result.query_type == QueryType.GRAPH
        assert result.confidence > 0.8
    
    def test_classify_relational(self, classifier):
        result = classifier.classify("ç»è®¡é¡¹ç®æ°é")
        
        assert result.query_type == QueryType.RELATIONAL
        assert result.confidence > 0.8
    
    def test_classify_semantic(self, classifier):
        result = classifier.classify("å¦ä½è¿è¡ä»£ç å®¡æ¥ï¼")
        
        assert result.query_type == QueryType.SEMANTIC_DOC
        assert result.confidence > 0.7
    
    def test_extract_entities(self, classifier):
        result = classifier.classify("æ¥æ¾ DOC-2024 å?REQ-123")
        
        assert len(result.entities) >= 2
        assert "DOC-2024" in result.entities
        assert "REQ-123" in result.entities
    
    def test_extract_keywords(self, classifier):
        result = classifier.classify("å¦ä½è¿è¡ä»£ç å®¡æ¥")
        
        assert len(result.keywords) > 0


class TestRoutingRules:
    
    @pytest.fixture
    def rules(self):
        return RoutingRules()
    
    def test_route_exact_query(self, rules):
        classification = ClassificationResult(
            query_type=QueryType.FACTUAL_EXACT,
            confidence=0.95,
            features={},
            entities=["PRJ-123"],
            keywords=[]
        )
        
        plan = rules.route(classification)
        
        assert plan.primary == BackendType.SQL
        assert plan.fusion == "first_match"
    
    def test_route_semantic_query(self, rules):
        classification = ClassificationResult(
            query_type=QueryType.SEMANTIC_DOC,
            confidence=0.85,
            features={},
            entities=[],
            keywords=["å¦ä½", "ä»£ç "]
        )
        
        plan = rules.route(classification)
        
        assert plan.primary == BackendType.VECTOR
        assert plan.fusion == "rrf"
    
    def test_route_graph_query(self, rules):
        classification = ClassificationResult(
            query_type=QueryType.GRAPH,
            confidence=0.90,
            features={},
            entities=[],
            keywords=["依赖"],
        )
        
        plan = rules.route(classification)
        
        assert plan.primary == BackendType.GRAPH
    
    def test_get_supported_backends(self, rules):
        backends = rules.get_supported_backends()
        
        assert BackendType.REDIS_CACHE in backends
        assert BackendType.VECTOR in backends


class TestFusionStrategies:
    
    @pytest.fixture
    def sample_evidences(self):
        return [
            [
                Evidence(id="doc1", content="content1", score=0.9, source_type="vector", metadata={}),
                Evidence(id="doc2", content="content2", score=0.8, source_type="vector", metadata={}),
            ],
            [
                Evidence(id="doc2", content="content2", score=0.95, source_type="sql", metadata={}),
                Evidence(id="doc3", content="content3", score=0.7, source_type="sql", metadata={}),
            ],
        ]
    
    def test_rrf_fusion(self, sample_evidences):
        fusion = RRFFusion(k=60)
        result = fusion.fuse(sample_evidences)
        
        assert len(result) == 3
        assert all("rrf_score" in e.metadata for e in result)
    
    def test_weighted_fusion(self, sample_evidences):
        fusion = WeightedFusion(normalize=True)
        result = fusion.fuse(sample_evidences, weights=[0.6, 0.4])
        
        assert len(result) == 3
    
    def test_cascade_fusion(self, sample_evidences):
        fusion = CascadeFusion(deduplicate=True)
        result = fusion.fuse(sample_evidences)
        
        ids = [e.id for e in result]
        assert len(ids) == len(set(ids))
    
    def test_first_match_fusion(self, sample_evidences):
        fusion = FusionFactory.create("first_match")
        result = fusion.fuse(sample_evidences)
        
        assert len(result) == 2
        assert all(e.source_type == "vector" for e in result)


class TestChunkers:
    
    @pytest.fixture
    def sample_text(self):
        return "第一段。\n\n第二段，比较长一些，包含更多内容。\n\n第三段。"

    def test_recursive_chunker(self, sample_text):
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk(sample_text, "test_doc")
        
        assert len(chunks) > 0
        assert all(c.doc_id == "test_doc" for c in chunks)
        assert all(len(c.content) <= 35 for c in chunks)
    
    def test_markdown_chunker(self):
        md_text = """Recovered docstring."""
        chunker = MarkdownChunker(chunk_size=100)
        chunks = chunker.chunk(md_text, "test_doc")
        
        assert len(chunks) >= 2
        assert any("æ é¢ä¸" in c.content or "æ é¢ä¸" in c.metadata.get("header", "") for c in chunks)
    
    def test_create_chunker(self):
        recursive = create_chunker(ChunkStrategy.RECURSIVE, chunk_size=300)
        assert isinstance(recursive, RecursiveChunker)
        
        markdown = create_chunker(ChunkStrategy.MARKDOWN)
        assert isinstance(markdown, MarkdownChunker)


class TestEmbedder:
    
    def test_embedder_initialization(self):
        class MockModel:
            def embed(self, texts):
                return [[0.1, 0.2, 0.3] for _ in texts]
        
        embedder = Embedder(
            embedding_model=MockModel(),
            batch_size=16,
            model_version="v1"
        )
        
        assert embedder._batch_size == 16
        assert embedder._model_version == "v1"


class TestHyDERewriter:
    
    def test_hyde_initialization(self):
        class MockLLM:
            def invoke(self, prompt):
                return type('Response', (), {'content': 'åè®¾ææ¡£åå®¹'})()
        
        rewriter = HyDERewriter(llm_model=MockLLM())
        
        assert rewriter._num_hypotheses == 1


class TestMultiQueryExpander:
    
    def test_multi_query_initialization(self):
        class MockLLM:
            def invoke(self, prompt):
                return type('Response', (), {'content': 'æ¥è¯¢1\næ¥è¯¢2\næ¥è¯¢3'})()
        
        expander = MultiQueryExpander(llm_model=MockLLM(), num_queries=3)
        
        assert expander._num_queries == 3


class TestQueryRewriterPipeline:
    
    def test_pipeline_disabled(self):
        pipeline = QueryRewriterPipeline(
            enable_hyde=False,
            enable_multi_query=False
        )
        
        assert not pipeline.is_enabled()
    
    def test_pipeline_enabled(self):
        pipeline = QueryRewriterPipeline(
            enable_hyde=True,
            enable_multi_query=True
        )
        
        assert pipeline.is_enabled()


class TestRetrievalPlan:
    
    def test_plan_to_dict(self):
        plan = RetrievalPlan(
            primary=BackendType.VECTOR,
            secondary=[BackendType.SQL],
            fusion="rrf",
            top_k=10,
        )
        
        result = plan.to_dict()
        
        assert result["primary"] == "vector"
        assert "sql" in result["secondary"]
        assert result["fusion"] == "rrf"
        assert result["top_k"] == 10
