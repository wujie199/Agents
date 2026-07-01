"""context_factory RAG tenant 解析。"""

from __future__ import annotations

from core.domain.context import RequestContext

from app.agents.context_factory import resolve_rag_tenant_id


def test_resolve_rag_tenant_dev_aligns_memory_tenant(tmp_path):
    req = RequestContext(
        tenant_id="tenant1",
        user_id="u1",
        session_id="s1",
        trace_id="t",
        channel="test",
    )
    empty = tmp_path / "data"
    empty.mkdir()
    assert (
        resolve_rag_tenant_id(req, profile="dev", data_dir=str(empty))
        == "tenant1"
    )


def test_resolve_rag_tenant_dev_legacy_chroma_fallback(tmp_path, monkeypatch):
    req = RequestContext(
        tenant_id="tenant1",
        user_id="u1",
        session_id="s1",
        trace_id="t",
        channel="test",
    )
    offline = tmp_path / "data" / "rag_offline" / "chroma_dev"
    offline.mkdir(parents=True)
    (offline / "marker").write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert resolve_rag_tenant_id(req, profile="dev", data_dir="data") == "default"


def test_resolve_rag_tenant_prefers_manifest_tenant_over_legacy(tmp_path, monkeypatch):
    from document.rag.application.indexing.index_manifest import IndexManifest

    req = RequestContext(
        tenant_id="tenant1",
        user_id="u1",
        session_id="s1",
        trace_id="t",
        channel="test",
    )
    rag_offline = tmp_path / "data" / "rag_offline"
    chroma = rag_offline / "chroma_dev"
    chroma.mkdir(parents=True)
    (chroma / "marker").write_text("x", encoding="utf-8")
    manifest = IndexManifest.for_data_dir(rag_offline)
    manifest.register(
        "tenant1",
        "abc123",
        doc_id="doc_abc",
        source_path="/tmp/a.pdf",
        model_version="v1",
        config_hash="hash1",
    )
    monkeypatch.chdir(tmp_path)
    assert resolve_rag_tenant_id(req, profile="dev", data_dir="data") == "tenant1"


def test_resolve_rag_tenant_production_memory():
    req = RequestContext(
        tenant_id="tenant1",
        user_id="u1",
        session_id="s1",
        trace_id="t",
        channel="test",
    )
    assert resolve_rag_tenant_id(req, profile="production") == "tenant1"
