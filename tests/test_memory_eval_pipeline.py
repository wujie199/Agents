# -*- coding: utf-8
"""Memory evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from document.memory.evaluation.dataset import load_memory_eval_dataset
from document.memory.evaluation.metrics import keyword_recall, kv_match_score
from document.memory.evaluation.pipeline import (
    eval_l1_extract_row,
    eval_session_search_row,
)


def test_keyword_recall():
    assert keyword_recall("维护周期建议 3 个月", ["维护", "周期"]) == 1.0
    assert keyword_recall("hello", ["维护"]) == 0.0


def test_kv_match_score():
    assert kv_match_score({"语言": "中文"}, {"语言": "中文"}) == 1.0
    assert kv_match_score({"语言": "中文简体"}, {"语言": "中文"}) == 1.0


def test_load_golden_sample():
    path = Path("data/memory_eval/golden/sample.jsonl")
    samples = load_memory_eval_dataset(path)
    assert len(samples) >= 2
    kinds = {s.kind for s in samples}
    assert "session_search" in kinds
    assert "l1_extract" in kinds


@pytest.mark.asyncio
async def test_eval_session_search_row():
    sample = load_memory_eval_dataset(
        "data/memory_eval/golden/sample.jsonl"
    )[0]
    memory = MagicMock()
    memory.ensure_session = AsyncMock()
    memory.persist_turn = AsyncMock()
    memory.session_search = AsyncMock(
        return_value="维护周期建议每 3 个月清理尘盒"
    )
    row = await eval_session_search_row(memory, sample, min_score=0.5)
    assert row.passed
    assert row.score >= 0.5


@pytest.mark.asyncio
async def test_eval_l1_extract_row_mock():
    from core.composition.run_context import RunContext
    from core.domain.context import RequestContext
    from app.agents.orchestration.chat_config import ChatAgentConfig
    from document.memory.evaluation.pipeline import run_memory_eval

    sample_path = "data/memory_eval/golden/sample.jsonl"
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        )
    )
    result = await run_memory_eval(
        memory=MagicMock(),
        ctx=ctx,
        dataset_path=sample_path,
        sample_limit=2,
    )
    l1_rows = [r for r in result["rows"] if r.get("kind") == "l1_extract"]
    assert l1_rows and l1_rows[0]["passed"]
