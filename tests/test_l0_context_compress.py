# -*- coding: utf-8 -*-
"""L0 上下文压缩 Phase A 测试。"""

from __future__ import annotations

import pytest

from core.domain.context import RequestContext

from agent_platform.memory.adapters.context_compressor import (
    CompressorConfig,
    HermesContextCompressor,
    compressor_config_from_dict,
)
from agent_platform.memory.adapters.context_window_manager import (
    ContextWindowManager,
    prune_old_tool_results,
    repair_tool_message_pairs,
)
from app.agents.memory.l0_context import (
    apply_l0_to_turn_messages,
    clear_l0_state,
    maybe_compress_turn_context,
)
from core.composition.run_context import RunContext


def _ctx(session: str = "s1") -> RequestContext:
    return RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id=session,
        trace_id="tr1",
        channel="test",
    )


def _long_text(n: int = 400) -> str:
    return "x" * n


def test_compressor_config_defaults():
    cfg = compressor_config_from_dict({})
    assert cfg.trigger_pct == 0.50
    assert cfg.compress_target_ratio == 0.20


@pytest.mark.asyncio
async def test_no_compress_below_threshold():
    comp = HermesContextCompressor(
        config=CompressorConfig(
            trigger_pct=0.5,
            context_window_tokens=1000,
            enabled=True,
        )
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    result = await comp.maybe_compress(messages, _ctx(), model_window=1000)
    assert result.savings_pct == 0.0
    assert result.compressed_messages is messages
    assert result.triggered is False


@pytest.mark.asyncio
async def test_compress_uses_prompt_tokens_when_provided(monkeypatch):
    comp = HermesContextCompressor(
        config=CompressorConfig(
            trigger_pct=0.5,
            context_window_tokens=1000,
            compress_target_ratio=0.2,
            anti_jitter_pct=0.01,
            enabled=True,
        )
    )

    async def _short_summary(head, context, **kwargs):
        return "[Context Summary]\ncompressed"

    monkeypatch.setattr(comp, "_generate_summary", _short_summary)

    messages = [
        {"role": "system", "content": "sys prefix"},
        {"role": "user", "content": "first question"},
        *[
            {"role": "user" if i % 2 == 0 else "assistant", "content": _long_text(300)}
            for i in range(12)
        ],
    ]
    result = await comp.maybe_compress(
        messages,
        _ctx(),
        model_window=1000,
        prompt_tokens=600,
    )
    assert result.triggered is True
    assert result.savings_pct >= 0.01
    assert any("[Context Summary" in str(m.get("content", "")) for m in result.compressed_messages)


@pytest.mark.asyncio
async def test_head_protection_keeps_system_and_first_user():
    comp = HermesContextCompressor(
        config=CompressorConfig(
            trigger_pct=0.1,
            context_window_tokens=500,
            compress_target_ratio=0.3,
            anti_jitter_pct=0.01,
            enabled=True,
        )
    )
    messages = [
        {"role": "system", "content": "L1 snapshot"},
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": _long_text(200)},
        {"role": "user", "content": _long_text(200)},
        {"role": "assistant", "content": _long_text(200)},
    ]
    result = await comp.maybe_compress(
        messages,
        _ctx(),
        model_window=500,
        prompt_tokens=200,
    )
    out = result.compressed_messages
    assert out[0]["content"] == "L1 snapshot"
    assert out[1]["content"] == "first user"


def test_prune_old_tool_results_outside_tail():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "tool", "content": _long_text(250), "tool_call_id": "a"},
        {"role": "tool", "content": _long_text(250), "tool_call_id": "b"},
    ]
    pruned, count = prune_old_tool_results(messages, tail_start=2, min_chars=200)
    assert count == 1
    assert pruned[1]["content"] == "[tool result truncated]"
    assert len(pruned[2]["content"]) > 200


def test_repair_tool_message_pairs_adds_missing_results():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "function": {"name": "search", "arguments": "{}"}}],
        },
    ]
    repaired = repair_tool_message_pairs(messages)
    assert len(repaired) == 2
    assert repaired[1]["role"] == "tool"
    assert repaired[1]["tool_call_id"] == "call-1"


def test_repair_tool_message_pairs_drops_orphan_tools():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "orphan", "tool_call_id": "x"},
    ]
    repaired = repair_tool_message_pairs(messages)
    assert len(repaired) == 1


@pytest.mark.asyncio
async def test_anti_jitter_rollback(monkeypatch):
    comp = HermesContextCompressor(
        config=CompressorConfig(
            trigger_pct=0.1,
            context_window_tokens=1000,
            compress_target_ratio=0.9,
            anti_jitter_pct=0.99,
            enabled=True,
        )
    )

    async def _tiny_summary(head, context, **kwargs):
        return "[Context Summary]\nshort"

    monkeypatch.setattr(comp, "_generate_summary", _tiny_summary)

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        *[{"role": "assistant", "content": _long_text(500)} for _ in range(6)],
    ]
    result = await comp.maybe_compress(
        messages,
        _ctx(),
        model_window=1000,
        prompt_tokens=500,
    )
    assert result.compressed_messages is messages
    assert result.savings_pct == 0.0


def test_context_window_manager_zones():
    mgr = ContextWindowManager(lambda t: max(1, len(t) // 4))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a" * 80},
        {"role": "user", "content": "b" * 80},
    ]
    zones = mgr.split_zones(messages, tail_token_budget=30)
    assert zones.protected_head[0]["role"] == "system"
    assert zones.protected_head[1]["content"] == "first"


def test_should_compress_uses_threshold():
    mgr = ContextWindowManager(lambda t: len(t))
    assert mgr.should_compress(500, 1000, trigger_pct=0.5) is True
    assert mgr.should_compress(499, 1000, trigger_pct=0.5) is False


def test_store_prompt_tokens_from_response():
    from app.agents.memory.l0_context import (
        pop_last_prompt_tokens,
        store_prompt_tokens_from_response,
    )

    run = RunContext(request=_ctx())
    run.extra = {}

    class _Usage:
        prompt_tokens = 1234

    class _Resp:
        usage = _Usage()

    store_prompt_tokens_from_response(run, _Resp())
    assert pop_last_prompt_tokens(run) == 1234
    assert pop_last_prompt_tokens(run) is None


@pytest.mark.asyncio
async def test_apply_l0_to_turn_messages_replaces_history():
    run = RunContext(request=_ctx())
    run.extra = {
        "context_compressor": HermesContextCompressor(
            config=CompressorConfig(enabled=True)
        ),
        "_l0_context_state": {
            "memory_snapshot_hash": "hash1",
            "summary_message": {"role": "system", "content": "[Context Summary]\nx"},
            "tail_messages": [{"role": "assistant", "content": "recent"}],
        },
    }
    fresh = [
        {"role": "system", "content": "fresh L1"},
        {"role": "user", "content": "old hist user"},
        {"role": "assistant", "content": "old hist asst"},
        {"role": "user", "content": "new question"},
    ]
    out = apply_l0_to_turn_messages(run, fresh, memory_snapshot_hash="hash1")
    assert out[0]["content"] == "fresh L1"
    assert out[1]["content"] == "[Context Summary]\nx"
    assert out[2]["content"] == "recent"
    assert out[-1]["content"] == "new question"


@pytest.mark.asyncio
async def test_l0_state_invalidated_on_l1_hash_change():
    run = RunContext(request=_ctx())
    run.extra = {
        "context_compressor": HermesContextCompressor(
            config=CompressorConfig(enabled=True)
        ),
        "_l0_context_state": {
            "memory_snapshot_hash": "old",
            "tail_messages": [{"role": "assistant", "content": "t"}],
        },
    }
    fresh = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    out = apply_l0_to_turn_messages(run, fresh, memory_snapshot_hash="new")
    assert len(out) == 2
    assert "_l0_context_state" not in run.extra


@pytest.mark.asyncio
async def test_maybe_compress_turn_context_stores_state(monkeypatch):
    run = RunContext(request=_ctx())
    comp = HermesContextCompressor(
        config=CompressorConfig(
            trigger_pct=0.1,
            context_window_tokens=800,
            compress_target_ratio=0.2,
            anti_jitter_pct=0.01,
            enabled=True,
        )
    )

    async def _short_summary(head, context, **kwargs):
        return "[Context Summary]\nok"

    monkeypatch.setattr(comp, "_generate_summary", _short_summary)
    run.extra = {"context_compressor": comp}
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        *[{"role": "assistant", "content": _long_text(400)} for _ in range(8)],
        {"role": "assistant", "content": "latest"},
    ]
    await maybe_compress_turn_context(
        run,
        messages,
        memory_snapshot_hash="h1",
        prompt_tokens=300,
        model_window=800,
    )
    assert "_l0_context_state" in run.extra
    assert run.extra.get("_l0_compress_count") == 1
    clear_l0_state(run)
    assert "_l0_context_state" not in run.extra


def test_resolve_model_window_tokens_from_registry():
    from agent_platform.model.registry import ModelRegistry

    from app.agents.memory.l0_context import resolve_model_window_tokens
    from app.agents.orchestration.chat_config import ChatAgentConfig

    class _Models:
        def get_context_window_tokens(self, role):
            return 64000 if role == "main_llm" else None

    run = RunContext(request=_ctx(), models=_Models())
    cfg = ChatAgentConfig(context_window_tokens=128000)
    assert resolve_model_window_tokens(run, cfg) == 64000


def test_resolve_model_window_tokens_falls_back_to_chat_cfg():
    from app.agents.memory.l0_context import resolve_model_window_tokens
    from app.agents.orchestration.chat_config import ChatAgentConfig

    run = RunContext(request=_ctx())
    cfg = ChatAgentConfig(context_window_tokens=96000)
    assert resolve_model_window_tokens(run, cfg) == 96000
