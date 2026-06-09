"""context_factory RAG tenant 解析。"""

from __future__ import annotations

from core.domain.context import RequestContext

from app.agents.context_factory import resolve_rag_tenant_id


def test_resolve_rag_tenant_dev_default():
    req = RequestContext(
        tenant_id="tenant1",
        user_id="u1",
        session_id="s1",
        trace_id="t",
        channel="test",
    )
    assert resolve_rag_tenant_id(req, profile="dev") == "default"


def test_resolve_rag_tenant_production_memory():
    req = RequestContext(
        tenant_id="tenant1",
        user_id="u1",
        session_id="s1",
        trace_id="t",
        channel="test",
    )
    assert resolve_rag_tenant_id(req, profile="production") == "tenant1"
