# -*- coding: utf-8 -*-
"""RAG query 改写与维护意图路由测试。"""

import pytest

from core.domain.evidence import Evidence, SourceType
from document.rag.application.retrieval.query_intent import (
    apply_post_rerank_maintenance_routing,
    boost_maintenance_evidences,
    demote_faq_template_evidences,
    detect_maintenance_intent,
    is_faq_source,
    is_maintenance_source,
    resolve_retrieval_tags,
    strip_query_templates,
)
from document.rag.application.retrieval.rewrite.rewrite_profile import (
    resolve_rewrite_profile,
)
from document.rag.application.retrieval.rewrite.retrieval_quality import (
    retrieval_satisfactory,
)
from document.rag.config.rewrite import RewriteConfig, TwoStageConfig
from document.rag.application.retrieval.rewrite.rule_based import RuleBasedQueryRewriter
from document.rag.application.retrieval.rewrite.combined import CombinedQueryRewriter
from document.rag.application.retrieval.rewrite.multi_query import (
    MultiQueryExpander,
    QueryRewriterPipeline,
)


def test_strip_query_templates():
    q = "清洁保养扫地机器人机身时应注意什么？"
    assert strip_query_templates(q) == "清洁保养扫地机器人机身时"


def test_detect_maintenance_intent():
    assert detect_maintenance_intent("清洁保养扫地机器人机身时应注意什么？")
    assert not detect_maintenance_intent("卫生间地面清扫需要注意什么")


def test_resolve_retrieval_tags_maintenance():
    tags, match = resolve_retrieval_tags("清洁机器人机身保养")
    assert "maintenance" in tags
    assert match == "any"


@pytest.mark.asyncio
async def test_rule_based_rewrite_expands_maintenance_query():
    rw = RuleBasedQueryRewriter(max_queries=4)
    out = await rw.rewrite("清洁保养扫地机器人机身时应注意什么？")
    assert out[0].startswith("清洁保养")
    assert any("断开电源" in q for q in out)
    assert len(out) >= 2


def test_boost_maintenance_evidences():
    maint = Evidence(
        id="m1",
        content="8. 清洁机器人时，全程断开电源",
        source_type=SourceType.VECTOR,
        score=0.5,
        metadata={"source_path": "/data/test_docs/维护保养.txt"},
    )
    faq = Evidence(
        id="f1",
        content="74. 卫生间地面清扫需要注意什么",
        source_type=SourceType.VECTOR,
        score=0.6,
        metadata={"source_path": "/data/test_docs/扫地机器人100问2.txt"},
    )
    query = "清洁保养扫地机器人机身时应注意什么？"
    boosted = boost_maintenance_evidences([faq, maint], query, boost=0.15)
    maint_ev = next(e for e in boosted if e.id == "m1")
    assert maint_ev.score == pytest.approx(0.65)
    assert boosted[0].id == "m1"


@pytest.mark.asyncio
async def test_combined_llm_rewrite_once_not_per_rule_query():
    """LLM multi-query 只对原始 query 调用一次，不对每条规则子 query 重复。"""
    calls: list[str] = []

    class TrackingLLM:
        async def ainvoke(self, messages, **kwargs):
            content = messages[0]["content"] if messages else ""
            calls.append(content)
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Msg",
                                    (),
                                    {"content": "LLM子查询A\nLLM子查询B"},
                                )()
                            },
                        )()
                    ]
                },
            )()

    rule = RuleBasedQueryRewriter(max_queries=4)
    llm_pipe = QueryRewriterPipeline(
        multi_query_expander=MultiQueryExpander(llm_model=TrackingLLM(), num_queries=2),
        enable_multi_query=True,
    )
    combined = CombinedQueryRewriter(
        rule_rewriter=rule,
        llm_pipeline=llm_pipe,
        llm_rewrite_once=True,
    )

    query = "清洁保养扫地机器人机身时应注意什么？"
    out = await combined.rewrite(query, llm_mode="always")

    assert len(calls) == 1
    assert query in calls[0]
    assert query in out
    assert any("断开电源" in q for q in out)
    assert any("LLM子查询A" in q for q in out)


@pytest.mark.asyncio
async def test_rewrite_queries_cap():
    from document.rag.application.retrieval.hybrid_pipeline import _rewrite_queries

    class ManyRewriter:
        def is_enabled(self):
            return True

        async def rewrite(self, query: str):
            return [f"q{i}" for i in range(10)]

    capped = await _rewrite_queries("orig", ManyRewriter(), max_queries=6)
    assert len(capped) == 6
    assert capped[0] == "q0"


def test_resolve_rewrite_profile_maintenance():
    assert resolve_rewrite_profile("清洁保养扫地机器人机身时应注意什么？") == "maintenance"


def test_resolve_rewrite_profile_generic():
    assert resolve_rewrite_profile("扫地机器人有哪些品牌") == "generic_knowledge"


def test_resolve_rewrite_profile_faq_like():
    assert resolve_rewrite_profile("卫生间地面清扫需要注意什么") == "faq_like"


def test_retrieval_satisfactory_maintenance_source():
    maint = Evidence(
        id="m1",
        content="8. 清洁机器人时，全程断开电源",
        source_type=SourceType.VECTOR,
        score=0.9,
        metadata={"source_path": "/data/test_docs/维护保养.txt", "rerank_score": 0.9},
    )
    faq = Evidence(
        id="f1",
        content="74. 卫生间地面清扫需要注意什么",
        source_type=SourceType.VECTOR,
        score=0.99,
        metadata={"source_path": "/data/test_docs/100问.txt", "rerank_score": 0.99},
    )
    cfg = TwoStageConfig(enabled=True, top1_min_rerank=0.75, require_maintenance_source=True)
    q = "清洁保养扫地机器人机身时应注意什么？"
    assert retrieval_satisfactory(q, "maintenance", [maint], cfg)
    assert not retrieval_satisfactory(q, "maintenance", [faq], cfg)


def test_demote_faq_template_evidences():
    maint = Evidence(
        id="m1",
        content="8. 清洁机器人时，全程断开电源",
        source_type=SourceType.VECTOR,
        score=0.8,
        metadata={"source_path": "/data/test_docs/维护保养.txt"},
    )
    faq = Evidence(
        id="f1",
        content="74. 卫生间地面清扫需要注意什么",
        source_type=SourceType.VECTOR,
        score=0.99,
        metadata={"source_path": "/data/test_docs/100问.txt"},
    )
    q = "清洁保养扫地机器人机身时应注意什么？"
    out = demote_faq_template_evidences([faq, maint], q)
    assert out[0].id == "m1"
    assert out[1].id == "f1"


def test_is_faq_source():
    assert is_faq_source({"source_path": "/data/扫地机器人100问2.txt"}, "189. 机身清洁")
    assert is_faq_source({"source_path": "/data/test_docs/选购指南.txt"}, "189. 机身清洁")
    assert not is_faq_source({"source_path": "/data/维护保养.txt"}, "8. 断开电源")


def test_is_maintenance_source_ignores_content_rule_tag():
    """文档级 maintenance 标签不应把选购指南/100问当成维护保养来源。"""
    polluted = {
        "source_path": "/data/test_docs/选购指南.txt",
        "tags": ["maintenance", "维护保养", "file:选购指南"],
    }
    assert not is_maintenance_source(polluted)
    assert is_faq_source(polluted, "189. 机身清洁难度")
    assert is_maintenance_source(
        {"source_path": "/data/test_docs/维护保养.txt", "tags": ["maintenance"]}
    )
    assert is_maintenance_source({"tags": ["file:维护保养"]})


def test_post_rerank_penalizes_guide_with_polluted_maintenance_tag():
    maint = Evidence(
        id="m8",
        content="8. 清洁机器人时，全程断开电源，禁止水洗",
        source_type=SourceType.VECTOR,
        score=0.70,
        metadata={
            "source_path": "/data/test_docs/维护保养.txt",
            "rerank_score": 0.70,
        },
    )
    guide189 = Evidence(
        id="g189",
        content="189. 机身清洁难度机身表面光滑无死角用湿抹布即可清洁",
        source_type=SourceType.VECTOR,
        score=0.89,
        metadata={
            "source_path": "/data/test_docs/选购指南.txt",
            "tags": ["maintenance", "维护保养", "file:选购指南"],
            "rerank_score": 0.89,
        },
    )
    q = "清洁保养机身应注意什么"
    out = apply_post_rerank_maintenance_routing(
        [guide189, maint],
        q,
        maintenance_boost=0.18,
        faq_penalty=0.12,
    )
    assert out[0].id == "m8"
    assert out[0].metadata["routing_score"] == pytest.approx(0.88)
    assert out[1].metadata["routing_score"] == pytest.approx(0.77)


def test_post_rerank_maintenance_routing_flips_top1():
    """模拟 rerank 后 FAQ 字面分更高，routing 后 maintenance 应排第一。"""
    maint = Evidence(
        id="m8",
        content="8. 清洁机器人时，全程断开电源，禁止水洗",
        source_type=SourceType.VECTOR,
        score=0.68,
        metadata={
            "source_path": "/data/test_docs/维护保养.txt",
            "rerank_score": 0.68,
        },
    )
    faq189 = Evidence(
        id="f189",
        content="189. 机身清洁可用湿抹布",
        source_type=SourceType.VECTOR,
        score=0.71,
        metadata={
            "source_path": "/data/test_docs/扫地机器人100问2.txt",
            "rerank_score": 0.71,
        },
    )
    faq37 = Evidence(
        id="f37",
        content="37. 按键防水",
        source_type=SourceType.VECTOR,
        score=0.65,
        metadata={
            "source_path": "/data/test_docs/扫地机器人100问2.txt",
            "rerank_score": 0.65,
        },
    )
    q = "清洁保养机身应注意什么"
    out = apply_post_rerank_maintenance_routing(
        [faq189, maint, faq37],
        q,
        maintenance_boost=0.18,
        faq_penalty=0.12,
    )
    assert out[0].id == "m8"
    assert out[0].metadata["routing_score"] == pytest.approx(0.86)
    assert out[1].metadata["routing_score"] == pytest.approx(0.59)


@pytest.mark.asyncio
async def test_combined_stage1_skips_llm_for_generic():
    calls: list[str] = []

    class TrackingLLM:
        async def ainvoke(self, messages, **kwargs):
            calls.append("llm")
            return type(
                "Response",
                (),
                {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": "x"})()})()]},
            )()

    from document.rag.config.rewrite import RewriteConfig, TwoStageConfig, TwoStageConfig

    rule = RuleBasedQueryRewriter(max_queries=4)
    llm_pipe = QueryRewriterPipeline(
        multi_query_expander=MultiQueryExpander(llm_model=TrackingLLM(), num_queries=2),
        enable_multi_query=True,
    )
    combined = CombinedQueryRewriter(
        rule_rewriter=rule,
        llm_pipeline=llm_pipe,
        rewrite_config=RewriteConfig(),
    )
    out = await combined.rewrite("扫地机器人有哪些品牌")
    assert calls == []
    assert out == ["扫地机器人有哪些品牌"]
