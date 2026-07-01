# -*- coding: utf-8 -*-
"""Tests for RAG evaluation pipeline (mocked LLM/RAG, no ragas API keys required)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.domain.context import RequestContext
from core.domain.evidence import DegradedReason, Evidence, EvidenceBundle, SourceType
from core.composition.run_context import RunContext

from document.rag.evaluation.dataset import (
    DatasetValidationError,
    EvalSample,
    load_eval_dataset,
)
from document.rag.evaluation.metrics import check_ragas_import, metric_names_for_mode
from document.rag.evaluation.pipeline import (
    PipelineRow,
    generate_answer,
    run_pipeline_sample,
)
from document.rag.evaluation.report import build_summary, render_report_md, write_run_reports
from document.rag.evaluation.stack_factory import (
    build_eval_run_context,
    load_eval_config,
)


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "golden.jsonl"
    rows = [
        {
            "id": "t1",
            "question": "问题一？",
            "ground_truth": "答案一",
            "reference_contexts": ["参考片段"],
            "tenant_id": "default",
        },
        {
            "id": "t2",
            "question": "问题二？",
            "ground_truth": "答案二",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_eval_dataset(sample_jsonl: Path):
    samples = load_eval_dataset(sample_jsonl)
    assert len(samples) == 2
    assert samples[0].id == "t1"
    assert samples[0].reference_contexts == ["参考片段"]


def test_load_eval_dataset_rejects_missing_ground_truth(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"id": "x", "question": "q?"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError):
        load_eval_dataset(path)


def test_metric_names_for_mode():
    cfg = load_eval_config("config")
    full = metric_names_for_mode("full", cfg)
    assert "faithfulness" in full
    retrieval = metric_names_for_mode("retrieval_only", cfg)
    assert "context_precision" in retrieval
    assert "faithfulness" not in retrieval


@pytest.mark.asyncio
async def test_run_pipeline_sample_full_mode():
    bundle = EvidenceBundle(
        evidences=[
            Evidence(
                id="e1",
                content="检索到的内容片段",
                source_type=SourceType.VECTOR,
                score=0.9,
            )
        ],
        empty=False,
    )
    mock_rag = MagicMock()
    mock_rag.route_and_retrieve = AsyncMock(return_value=bundle)

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "生成的回答"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    mock_models = MagicMock()
    mock_models.get_model = MagicMock(return_value=mock_llm)

    request = RequestContext(
        tenant_id="default",
        user_id="u",
        session_id="s",
        trace_id="t",
        channel="test",
    )
    ctx = RunContext(request=request, rag=mock_rag, models=mock_models, extra={"data_dir": "data"})
    sample = EvalSample(id="s1", question="测试问题？", ground_truth="标准答案")
    eval_cfg = load_eval_config("config")

    row = await run_pipeline_sample(
        ctx,
        sample,
        mode="full",
        eval_cfg=eval_cfg,
        profile="dev",
        data_dir="data",
    )
    assert row.answer == "生成的回答"
    assert row.contexts == ["检索到的内容片段"]
    assert row.evidence_count == 1
    assert row.retrieval_empty is False
    mock_rag.route_and_retrieve.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_pipeline_sample_retrieval_only():
    bundle = EvidenceBundle.empty_bundle(
        DegradedReason.PARTIAL_RESULTS, "no_hits", plan={"mode": "vector"}
    )
    mock_rag = MagicMock()
    mock_rag.route_and_retrieve = AsyncMock(return_value=bundle)

    request = RequestContext(
        tenant_id="default",
        user_id="u",
        session_id="s",
        trace_id="t",
        channel="test",
    )
    ctx = RunContext(request=request, rag=mock_rag, extra={"data_dir": "data"})
    sample = EvalSample(id="s2", question="空检索？", ground_truth="答案")
    eval_cfg = load_eval_config("config")

    row = await run_pipeline_sample(
        ctx,
        sample,
        mode="retrieval_only",
        eval_cfg=eval_cfg,
        profile="dev",
        data_dir="data",
    )
    assert row.answer == ""
    assert row.retrieval_empty is True
    assert row.retrieval_degraded is True


@pytest.mark.asyncio
async def test_generate_answer_uses_prompt_builder():
    bundle = EvidenceBundle(
        evidences=[
            Evidence(
                id="e1",
                content="证据文本",
                source_type=SourceType.VECTOR,
                score=0.8,
            )
        ],
        empty=False,
    )
    captured: dict = {}

    async def _fake_ainvoke(messages):
        captured["messages"] = messages
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "OK"
        return resp

    mock_llm = MagicMock()
    mock_llm.ainvoke = _fake_ainvoke
    mock_models = MagicMock()
    mock_models.get_model = MagicMock(return_value=mock_llm)

    ctx = RunContext(
        request=RequestContext(
            tenant_id="t",
            user_id="u",
            session_id="s",
            trace_id="tr",
            channel="test",
        ),
        models=mock_models,
    )
    sample = EvalSample(id="x", question="用户问题", ground_truth="gt")
    text, ms = await generate_answer(ctx, sample, bundle, load_eval_config("config"))
    assert text == "OK"
    assert ms >= 0
    assert captured["messages"][0]["role"] == "system"
    assert "证据文本" in captured["messages"][0]["content"]
    assert captured["messages"][-1]["content"] == "用户问题"


def test_build_eval_run_context_disables_cache():
    request = RequestContext(
        tenant_id="default",
        user_id="u",
        session_id="s",
        trace_id="t",
        channel="test",
    )
    with patch("document.rag.evaluation.stack_factory.build_chat_run_context") as mock_build:
        mock_rag = MagicMock()
        mock_rag._enable_cache = True
        mock_rag._router = MagicMock()
        mock_rag._router._enable_cache = True
        mock_build.return_value = RunContext(
            request=request,
            rag=mock_rag,
            extra={"data_dir": "data"},
        )
        ctx = build_eval_run_context(request, profile="dev", data_dir="data")
        assert mock_rag._enable_cache is False
        assert mock_rag._router._enable_cache is False
        assert ctx.extra.get("rag_tenant_id") is not None


def test_report_writers(tmp_path: Path):
    rows = [
        PipelineRow(
            id="r1",
            question="q",
            ground_truth="gt",
            answer="a",
            contexts=["c"],
            retrieval_ms=12.5,
            generation_ms=34.0,
        )
    ]
    summary = build_summary(
        run_id="test",
        mode="full",
        profile="dev",
        dataset_path="data.jsonl",
        sample_count=1,
        rows=rows,
        metric_means={"faithfulness": 0.9},
        config_meta={"cache_disabled": True},
    )
    paths = write_run_reports(
        tmp_path,
        summary=summary,
        detail_rows=[rows[0].to_detail_dict()],
        failure_rows=[],
    )
    assert Path(paths["summary"]).is_file()
    assert Path(paths["report_html"]).is_file()
    md = render_report_md(summary)
    assert "faithfulness" in md


@pytest.mark.asyncio
async def test_run_rag_eval_skip_ragas(tmp_path: Path, sample_jsonl: Path):
    from document.rag.evaluation.run import run_rag_eval

    bundle = EvidenceBundle(
        evidences=[
            Evidence(
                id="e1",
                content="ctx",
                source_type=SourceType.VECTOR,
                score=0.5,
            )
        ],
        empty=False,
    )
    mock_rag = MagicMock()
    mock_rag.route_and_retrieve = AsyncMock(return_value=bundle)
    mock_rag._enable_cache = True

    mock_llm = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "ans"
    mock_llm.ainvoke = AsyncMock(return_value=resp)
    mock_models = MagicMock()
    mock_models.get_model = MagicMock(return_value=mock_llm)

    fake_ctx = RunContext(
        request=RequestContext(
            tenant_id="default",
            user_id="u",
            session_id="s",
            trace_id="t",
            channel="test",
        ),
        rag=mock_rag,
        models=mock_models,
        extra={"data_dir": "data", "rag_tenant_id": "default"},
    )

    from dataclasses import replace

    out_dir = tmp_path / "results" / "unit-run"
    eval_cfg = replace(
        load_eval_config("config"),
        output={"results_dir": str(tmp_path / "results"), "preview_chars": 100},
    )

    with patch("document.rag.evaluation.run.build_eval_run_context", return_value=fake_ctx):
        with patch("document.rag.evaluation.run.load_eval_config", return_value=eval_cfg):
            result = await run_rag_eval(
                profile="dev",
                dataset_path=str(sample_jsonl),
                mode="full",
                run_id="unit-run",
                sample_limit=1,
                skip_ragas=True,
            )

    assert result["run_id"] == "unit-run"
    assert Path(result["output_dir"]).is_dir()
    assert (Path(result["output_dir"]) / "summary.json").is_file()


@pytest.mark.skipif(
    check_ragas_import() is not None,
    reason="ragas not installed or langchain deps incompatible",
)
def test_build_ragas_metrics_when_installed():
    from document.rag.evaluation.metrics import build_ragas_metrics

    cfg = load_eval_config("config")
    metrics = build_ragas_metrics(
        "retrieval_only",
        cfg,
        llm=MagicMock(),
        embeddings=None,
    )
    assert len(metrics) >= 2
