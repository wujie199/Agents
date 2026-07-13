# -*- coding: utf-8
"""HITL / L4 / PostgresStore 企业评测测试。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from document.memory.evaluation.dataset import load_memory_eval_dataset
from document.memory.evaluation.metrics import user_snapshot_kv
from document.memory.evaluation.pipeline import (
    eval_hitl_finalize_row,
    eval_l4_merge_row,
    run_memory_eval,
)
from document.memory.evaluation.run import run_memory_eval_job
from document.memory.evaluation.stack_factory import build_eval_memory_stack


@pytest.mark.asyncio
async def test_hitl_finalize_golden(tmp_path):
    memory, archive_db, ctx, _ = await build_eval_memory_stack(tmp_root=tmp_path)
    try:
        sample = load_memory_eval_dataset(
            "data/memory_eval/golden/hitl_finalize.jsonl"
        )[0]
        row = await eval_hitl_finalize_row(memory, sample)
        assert row.passed, row.detail
    finally:
        await archive_db.close()


@pytest.mark.asyncio
async def test_l4_merge_golden(tmp_path):
    sample = load_memory_eval_dataset("data/memory_eval/golden/l4_merge.jsonl")[0]
    sub = tmp_path / "l4_sub"
    memory, archive_db, _, _ = await build_eval_memory_stack(
        tmp_root=sub,
        enable_l4=True,
        l4_facts=sample.l4_facts,
        tenant_id=sample.tenant_id,
        user_id=sample.user_id,
    )
    try:
        row = await eval_l4_merge_row(memory, sample)
        assert row.passed, row.detail
        detail = json.loads(row.detail)
        assert detail.get("l4_merged", 0) >= 1
    finally:
        await archive_db.close()


@pytest.mark.asyncio
async def test_run_hitl_and_l4_datasets(tmp_path):
    memory, archive_db, ctx, _ = await build_eval_memory_stack(tmp_root=tmp_path)

    async def l4_factory(sample):
        sub = tmp_path / f"l4_{sample.id}"
        mem, _, _, _ = await build_eval_memory_stack(
            tmp_root=sub,
            enable_l4=True,
            l4_facts=sample.l4_facts,
            tenant_id=sample.tenant_id,
            user_id=sample.user_id,
        )
        return mem

    try:
        for path in (
            "data/memory_eval/golden/hitl_finalize.jsonl",
            "data/memory_eval/golden/l4_merge.jsonl",
        ):
            result = await run_memory_eval(
                memory=memory,
                ctx=ctx,
                dataset_path=path,
                l4_memory_factory=l4_factory,
            )
            assert result["pass_rate"] == 1.0, path
    finally:
        await archive_db.close()


def test_user_snapshot_kv_parser():
    text = "# SYSTEM MEMORY\n\n\n# USER PREFERENCES\n\n称呼: 小王\n语言: 中文\n"
    kv = user_snapshot_kv(text)
    assert kv["称呼"] == "小王"
    assert kv["语言"] == "中文"


def test_langgraph_store_registry_inmemory():
    from agent_platform.memory.adapters.hot_memory_langgraph_store_adapter import (
        HotMemoryLangGraphStoreAdapter,
        build_langgraph_memory_store,
    )
    from core.domain.context import RequestContext
    from core.ports.memory import MemoryDelta

    store = build_langgraph_memory_store({})
    hot = HotMemoryLangGraphStoreAdapter(store)
    ctx = RequestContext(
        tenant_id="t1", user_id="u1", session_id="s1", trace_id="t", channel="test"
    )
    hot.apply_delta("t1", "u1", MemoryDelta(key="称呼", value="测试", source="user"))
    assert "测试" in hot.compose_snapshot(ctx).memory_text


@pytest.mark.asyncio
async def test_memory_eval_sample_smoke(tmp_path):
    memory, archive_db, ctx, _ = await build_eval_memory_stack(tmp_root=tmp_path)
    try:
        result = await run_memory_eval(
            memory=memory,
            ctx=ctx,
            dataset_path="data/memory_eval/golden/sample.jsonl",
            use_mock_only=True,
        )
        assert result["pass_rate"] > 0
        assert result["total"] >= 1
    finally:
        await archive_db.close()


@pytest.mark.asyncio
async def test_memory_eval_ci_smoke(tmp_path):
    result = await run_memory_eval_job(
        dataset_path="data/memory_eval/golden/sample.jsonl",
        run_id="ci-smoke",
        data_dir=str(tmp_path),
        use_mock_only=True,
    )
    assert result["pass_rate"] == 1.0
    assert (tmp_path / "memory_eval" / "results" / "ci-smoke" / "summary.json").is_file()


@pytest.mark.asyncio
async def test_l1_extract_mock_vs_llm_flag(tmp_path):
    memory, archive_db, ctx, _ = await build_eval_memory_stack(tmp_root=tmp_path)
    ctx.models = MagicMock()
    try:
        result = await run_memory_eval(
            memory=memory,
            ctx=ctx,
            dataset_path="data/memory_eval/golden/sample.jsonl",
            use_mock_only=True,
        )
        l1 = [r for r in result["rows"] if r["kind"] == "l1_extract"]
        assert l1 and l1[0]["passed"]
    finally:
        await archive_db.close()
