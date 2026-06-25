"""memory_metrics 单元测试。"""

from __future__ import annotations

from core.domain.context import RequestContext
from core.composition.run_context import RunContext

from agent_platform.infrastructure.observability.adapter import (
    ObservabilityPortAdapter,
)

from app.agents.memory.memory_metrics import (
    get_memory_metric_stats,
    get_memory_metric_stats_from_obs,
    prometheus_text,
    record_chat_turn,
    record_l1_confirm,
)


def _ctx_with_obs() -> RunContext:
    obs = ObservabilityPortAdapter(service_name="test")
    return RunContext(
        request=RequestContext(
            tenant_id="t",
            user_id="u",
            session_id="s",
            trace_id="tr",
            channel="test",
        ),
        observability=obs,
    )


def test_record_chat_turn_metrics():
    ctx = _ctx_with_obs()
    record_chat_turn(
        ctx,
        evidence_count=2,
        rag_empty=False,
        history_turns=3,
        engine="langgraph",
    )
    stats = get_memory_metric_stats(ctx)
    assert "memory.chat.turn" in stats
    assert stats["memory.chat.turn"]["count"] == 1
    assert stats["memory.chat.evidence_count"]["sum"] == 2.0


def test_prometheus_text_filters_memory_prefix():
    ctx = _ctx_with_obs()
    record_l1_confirm(ctx, 2)
    ctx.observability.record_metric("other.metric", 1.0)
    text = prometheus_text(ctx)
    assert "memory.l1.confirm_total" in text
    assert "other.metric" not in text


def test_stats_from_shared_observability():
    obs = ObservabilityPortAdapter(service_name="shared")
    obs.record_metric("memory.purge", 1.0)
    stats = get_memory_metric_stats_from_obs(obs)
    assert stats["memory.purge"]["count"] == 1
