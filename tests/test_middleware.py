# -*- coding: utf-8 -*-
"""Middleware 洋葱模型单元测试。"""

from __future__ import annotations

import asyncio
import pytest
from typing import Any, Dict, Optional

from app.agents.middleware import Middleware
from app.agents.middleware.compose import wrap_node, compose_middlewares
from app.agents.middleware.tracing import TracingMiddleware
from app.agents.middleware.request_context import RequestContextMiddleware
from app.agents.middleware.metrics import MetricsMiddleware
from app.agents.middleware.logging import LoggingMiddleware
from app.agents.middleware.policy import PolicyMiddleware
from app.agents.middleware.privacy import PrivacyMiddleware
from app.agents.middleware.audit import AuditMiddleware


class _SimpleMW:
    """测试用简单 middleware，记录 on_enter/on_exit 调用顺序。"""

    def __init__(self, name: str, log: list):
        self._name = name
        self._log = log

    @property
    def name(self) -> str:
        return self._name

    async def on_enter(self, node_name, state, config) -> Dict[str, Any]:
        self._log.append(f"{self._name}:enter")
        return {f"{self._name}_entered": True}

    async def on_exit(
        self, node_name, state, config, result, *,
        error=None, extra=None,
    ) -> None:
        self._log.append(f"{self._name}:exit")


class TestWrapNode:
    @pytest.mark.asyncio
    async def test_onion_order(self):
        """洋葱模型：enter 顺序 → node → exit 逆序。"""
        log = []
        mw1 = _SimpleMW("mw1", log)
        mw2 = _SimpleMW("mw2", log)

        async def node_fn(state, config):
            log.append("node")
            return {"result": "ok"}

        wrapped = await wrap_node([mw1, mw2], node_fn, "test_node")
        result = await wrapped({}, {})

        assert log == ["mw1:enter", "mw2:enter", "node", "mw2:exit", "mw1:exit"]
        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """节点异常应传播，但 on_exit 仍执行。"""
        log = []
        mw = _SimpleMW("mw", log)

        async def node_fn(state, config):
            raise ValueError("boom")

        wrapped = await wrap_node([mw], node_fn, "err_node")
        with pytest.raises(ValueError, match="boom"):
            await wrapped({}, {})
        assert "mw:exit" in log

    @pytest.mark.asyncio
    async def test_exit_error_swallowed(self):
        """on_exit 异常不应影响主流程。"""
        class BadExitMW:
            @property
            def name(self):
                return "bad_exit"

            async def on_enter(self, *a, **kw):
                return {}

            async def on_exit(self, *a, **kw):
                raise RuntimeError("exit error")

        async def node_fn(state, config):
            return {"ok": True}

        wrapped = await wrap_node([BadExitMW()], node_fn, "test")
        result = await wrapped({}, {})
        assert result == {"ok": True}


class TestComposeMiddlewares:
    def test_returns_list(self):
        mw1 = TracingMiddleware()
        mw2 = LoggingMiddleware()
        result = compose_middlewares(mw1, mw2)
        assert len(result) == 2
        assert result[0] is mw1


class TestTracingMiddleware:
    @pytest.mark.asyncio
    async def test_on_enter_returns_ids(self):
        mw = TracingMiddleware()
        ctx = await mw.on_enter("test_node", {}, {"configurable": {"trace_id": "t1"}})
        assert ctx["trace_id"] == "t1"
        assert "span_id" in ctx
        assert "_span_start" in ctx

    @pytest.mark.asyncio
    async def test_on_enter_generates_trace_id(self):
        mw = TracingMiddleware()
        ctx = await mw.on_enter("test_node", {}, {})
        assert ctx["trace_id"]
        assert len(ctx["trace_id"]) > 0

    @pytest.mark.asyncio
    async def test_on_enter_starts_observability_span(self):
        from agent_platform.infrastructure.observability.adapter import (
            ObservabilityPortAdapter,
        )
        from core.domain.context import RequestContext
        from core.composition.run_context import RunContext

        obs = ObservabilityPortAdapter(service_name="test-tracing")
        run_ctx = RunContext(
            request=RequestContext(
                tenant_id="t",
                user_id="u",
                session_id="s",
                trace_id="tr-1",
                channel="test",
            ),
            observability=obs,
        )
        config = {"configurable": {"trace_id": "tr-1", "run_ctx": run_ctx}}
        mw = TracingMiddleware()
        ctx = await mw.on_enter("prepare", {}, config)
        await mw.on_exit(
            "prepare", {}, config, {"ok": True}, extra=ctx
        )
        assert len(obs._spans) == 1
        span = list(obs._spans.values())[0]
        assert span.name == "graph.prepare"
        assert span.end_time is not None


class TestRequestContextMiddleware:
    @pytest.mark.asyncio
    async def test_propagates_trace_id_from_config(self):
        mw = RequestContextMiddleware()
        config = {"configurable": {"trace_id": "req-123"}}
        ctx = await mw.on_enter("prepare", {}, config)
        assert ctx["trace_id"] == "req-123"
        assert config["configurable"]["trace_id"] == "req-123"

    @pytest.mark.asyncio
    async def test_injects_identity_attributes(self):
        mw = RequestContextMiddleware()
        run_ctx = MagicMockRunCtx("tenant-a")
        run_ctx.request = type(
            "R",
            (),
            {
                "tenant_id": "tenant-a",
                "user_id": "user-b",
                "session_id": "sess-c",
                "trace_id": "tr",
            },
        )()
        config = {"configurable": {"run_ctx": run_ctx}}
        ctx = await mw.on_enter("agent", {}, config)
        assert ctx["span_attributes"]["tenant_id"] == "tenant-a"
        assert ctx["span_attributes"]["user_id"] == "user-b"
        assert ctx["span_attributes"]["session_id"] == "sess-c"


class TestMetricsMiddleware:
    @pytest.mark.asyncio
    async def test_records_graph_node_duration(self):
        from agent_platform.infrastructure.observability.adapter import (
            ObservabilityPortAdapter,
        )
        from core.domain.context import RequestContext
        from core.composition.run_context import RunContext

        obs = ObservabilityPortAdapter(service_name="test-metrics")
        run_ctx = RunContext(
            request=RequestContext(
                tenant_id="t-metrics",
                user_id="u",
                session_id="s",
                trace_id="tr",
                channel="test",
            ),
            observability=obs,
        )
        mw = MetricsMiddleware(node_thresholds={"prepare": 10})
        enter_ctx = await mw.on_enter("prepare", {}, {"configurable": {"run_ctx": run_ctx}})
        await mw.on_exit(
            "prepare",
            {},
            {"configurable": {"run_ctx": run_ctx}},
            {"ok": True},
            extra=enter_ctx,
        )
        metrics = obs.get_metrics()
        assert "graph.node.duration_ms" in metrics


class TestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_on_enter_returns_empty(self):
        mw = LoggingMiddleware()
        ctx = await mw.on_enter("test_node", {}, {})
        assert ctx == {}


class TestPolicyMiddleware:
    @pytest.mark.asyncio
    async def test_on_enter_no_tenant_ok(self):
        mw = PolicyMiddleware()
        ctx = await mw.on_enter("test_node", {}, {})
        assert ctx == {}

    @pytest.mark.asyncio
    async def test_qps_limit(self):
        mw = PolicyMiddleware(max_qps_per_tenant=2)
        config = {"configurable": {"run_ctx": MagicMockRunCtx("t1")}}
        # 前2次不报错
        await mw.on_enter("n", {}, config)
        await mw.on_enter("n", {}, config)
        # 第3次应报错
        with pytest.raises(PermissionError):
            await mw.on_enter("n", {}, config)


class MagicMockRunCtx:
    def __init__(self, tenant_id):
        self.request = type("R", (), {"tenant_id": tenant_id})()


class TestPrivacyMiddleware:
    @pytest.mark.asyncio
    async def test_detect_pii_phone(self):
        found = PrivacyMiddleware._detect_pii("我的手机号是13812345678")
        assert "phone" in found

    @pytest.mark.asyncio
    async def test_detect_pii_email(self):
        found = PrivacyMiddleware._detect_pii("联系 test@example.com")
        assert "email" in found

    @pytest.mark.asyncio
    async def test_mask_pii(self):
        text = "手机13812345678，邮箱test@example.com"
        masked = PrivacyMiddleware.mask_pii(text)
        assert "138****5678" in masked
        assert "te***@example.com" in masked

    @pytest.mark.asyncio
    async def test_disabled_no_detect(self):
        mw = PrivacyMiddleware(enabled=False)
        result = {"assistant_text": "手机13812345678"}
        # on_exit with enabled=False → no logging (just return)
        await mw.on_exit("n", {}, {}, result)


class TestAuditMiddleware:
    @pytest.mark.asyncio
    async def test_on_exit_logs_audit(self):
        mw = AuditMiddleware()
        result = {"assistant_text": "hello", "user_input": "hi"}
        # Should not raise
        await mw.on_exit(
            "test_node", {}, {}, result,
            extra={"trace_id": "t1", "span_id": "s1", "duration_ms": 42.0},
        )

    @pytest.mark.asyncio
    async def test_on_enter_returns_empty(self):
        mw = AuditMiddleware()
        ctx = await mw.on_enter("test_node", {}, {})
        assert ctx == {}
