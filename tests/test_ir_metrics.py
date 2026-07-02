# -*- coding: utf-8 -*-
"""Tests for IR retrieval metrics."""

from __future__ import annotations

from document.rag.evaluation.ir_metrics import (
    aggregate_ir_metrics,
    compute_sample_ir_scores,
    contexts_match,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from document.rag.evaluation.pipeline import PipelineRow


def test_contexts_match_faq_with_cleaning_artifact():
    from document.rag.evaluation.ir_metrics import contexts_match

    ref = "1. 首次使用扫地机器人需要做什么\n拆除机身所有包装配件，充满电后建图。"
    ret = "1. 首次使用扫地机器人需要做什么\n拆除机身所有包装配件充满电后在空旷环境下启动建图"
    assert contexts_match(ref, ret)


def test_contexts_match_maint_punctuation():
    ref = "3. 每日检查机器人底部驱动轮、万向轮，用镊子挑出缠绕的毛发、杂物"
    ret = "3. 每日检查机器人底部驱动轮万向轮用镊子挑出缠绕的毛发杂物"
    assert contexts_match(ref, ret)


def test_hit_mrr_ndcg_perfect_rank1():
    rels = [1, 0, 0, 0, 0]
    assert hit_at_k(rels, 1) == 1.0
    assert mrr_at_k(rels, 5) == 1.0
    assert ndcg_at_k(rels, 5) == 1.0


def test_hit_mrr_rank3():
    rels = [0, 0, 1, 0, 0]
    assert hit_at_k(rels, 5) == 1.0
    assert hit_at_k(rels, 2) == 0.0
    assert mrr_at_k(rels, 5) == 1 / 3


def test_recall_at_k_multi_reference():
    refs = ["ref A content here", "ref B content here"]
    retrieved = ["noise", "ref A content here extended", "other"]
    assert recall_at_k(refs, retrieved, 3, min_overlap_chars=10) == 0.5


def test_compute_sample_ir_scores():
    scores = compute_sample_ir_scores(
        reference_contexts=["标准 chunk 文本内容用于匹配"],
        retrieved_contexts=["无关", "标准 chunk 文本内容用于匹配检索"],
        ks=[1, 5],
        min_overlap_chars=10,
    )
    assert scores["hit@1"] == 0.0
    assert scores["hit@5"] == 1.0
    assert scores["mrr@5"] == 0.5


def test_aggregate_ir_metrics():
    rows = [
        PipelineRow(
            id="a",
            question="q",
            ground_truth="gt",
            reference_contexts=["alpha beta gamma delta"],
            contexts=["alpha beta gamma delta"],
        ),
        PipelineRow(
            id="b",
            question="q2",
            ground_truth="gt2",
            reference_contexts=["foo bar baz qux"],
            contexts=["miss"],
        ),
    ]
    means = aggregate_ir_metrics(rows, ks=[1, 5], min_overlap_chars=8)
    assert means["hit@1"] == 0.5
    assert rows[0].ir_scores["hit@1"] == 1.0
    assert rows[1].ir_scores["hit@1"] == 0.0
