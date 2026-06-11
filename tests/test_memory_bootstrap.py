"""记忆 dev bootstrap：默认全开、无需 MEMORY_CONFIG。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from core.domain.context import RequestContext
from core.composition.run_context import RunContext

from agent_platform.memory.adapters.config_loader import load_memory_config
from app.agents.context_factory import resolve_rag_tenant_id
from app.agents.chat_config import load_chat_config
from app.agents.memory_bootstrap import (
    bootstrap_memory_runtime,
    seed_l4_profile_if_missing,
)


def test_dev_chat_profile_relaxes_hitl():
    dev_cfg = load_chat_config("config", profile="dev")
    prod_cfg = load_chat_config("config", profile="production")
    assert dev_cfg.remember_require_hitl is False
    assert dev_cfg.auto_confirm_pending_on_exit is True
    assert prod_cfg.remember_require_hitl is True
    assert prod_cfg.auto_confirm_pending_on_exit is False


def test_default_memory_yml_enables_full_stack(monkeypatch):
    monkeypatch.delenv("MEMORY_CONFIG", raising=False)
    cfg = load_memory_config("config/memory.yml")
    assert cfg.get("enable_session_vector_index") is True
    assert cfg.get("enable_cold_archive") is True
    assert cfg.get("skill_auto_extract_draft") is True
    assert cfg.get("session_embedding_backend") == "mock"


def test_seed_l4_profile_if_missing(tmp_path):
    created = seed_l4_profile_if_missing(
        profiles_dir=tmp_path / "profiles",
        tenant_id="t1",
        user_id="u9",
    )
    assert created is True
    path = tmp_path / "profiles" / "t1" / "u9.yaml"
    assert path.is_file()
    assert seed_l4_profile_if_missing(
        profiles_dir=tmp_path / "profiles",
        tenant_id="t1",
        user_id="u9",
    ) is False


def test_resolve_rag_tenant_aligns_without_legacy_chroma(tmp_path):
    req = RequestContext(
        tenant_id="tenant1",
        user_id="u1",
        session_id="s1",
        trace_id="t",
        channel="test",
    )
    empty_data = tmp_path / "empty_data"
    empty_data.mkdir()
    assert (
        resolve_rag_tenant_id(req, profile="dev", data_dir=str(empty_data))
        == "tenant1"
    )


@pytest.mark.asyncio
async def test_bootstrap_memory_runtime(tmp_path):
    req = RequestContext(
        tenant_id="tenant1",
        user_id="user1",
        session_id="boot_sess",
        trace_id="t",
        channel="test",
    )
    memory = MagicMock()
    memory.ensure_session = AsyncMock()
    memory.reindex_session_vectors = AsyncMock(
        return_value={"indexed": 0, "errors": 0, "batches": 0}
    )
    ctx = RunContext(request=req, memory=memory, extra={})
    profiles = tmp_path / "external_profiles"
    report = await bootstrap_memory_runtime(
        ctx,
        data_dir=str(tmp_path),
        config_dir="config",
        profile="dev",
    )
    assert report.get("memory_ready") is True
    memory.ensure_session.assert_awaited_once()
    assert "vector_reindex" in report
