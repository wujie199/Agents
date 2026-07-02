# -*- coding: utf-8 -*-
"""Phase D：MemoryManager / MemoryProvider 测试。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
import yaml

from core.domain.context import RequestContext
from core.ports.memory.provider import BaseMemoryProvider, MemorySnippet

from agent_platform.memory.adapters.memory_fence import strip_memory_fences
from agent_platform.memory.adapters.memory_manager import MemoryManager
from agent_platform.memory.adapters.memory_provider_factory import (
    build_external_memory_provider,
    resolve_l4_provider_name,
)
from agent_platform.memory.adapters.memory_providers.file_profile_provider import (
    FileProfileProvider,
)
from agent_platform.memory.adapters.l1_memory_store import L1MemoryStore


def _ctx(session: str = "s1", user: str = "u1") -> RequestContext:
    return RequestContext(
        tenant_id="t1",
        user_id=user,
        session_id=session,
        trace_id="tr",
        channel="test",
    )


class SlowProvider(BaseMemoryProvider):
    def __init__(self, delay_s: float = 2.0):
        self._delay = delay_s

    @property
    def name(self) -> str:
        return "slow_external"

    async def prefetch_turn(self, user_message: str, context: RequestContext):
        await asyncio.sleep(self._delay)
        return [MemorySnippet(content="slow fact", source=self.name)]


class MirrorProvider(BaseMemoryProvider):
    def __init__(self):
        self.writes: list[tuple[str, str, str, dict]] = []

    @property
    def name(self) -> str:
        return "mirror"

    async def on_memory_write(
        self, tenant_id: str, user_id: str, action: str, payload: dict[str, Any]
    ) -> None:
        self.writes.append((tenant_id, user_id, action, payload))


@pytest.mark.asyncio
async def test_prefetch_timeout_fail_open():
    mgr = MemoryManager(
        external=SlowProvider(delay_s=2.5),
        prefetch_timeout_ms=200,
    )
    snippets = await mgr.prefetch_all("hello", _ctx())
    assert snippets == []


def test_register_single_external_provider_limit():
    mgr = MemoryManager()
    mgr.register_external(MirrorProvider())
    with pytest.raises(ValueError, match="already registered"):
        mgr.register_external(MirrorProvider())


def test_build_memory_context_block_and_fence_injection():
    mgr = MemoryManager()
    snippets = [
        MemorySnippet(content="Name: Alice", source="file_profile"),
        MemorySnippet(content="Lang: zh", source="file_profile"),
    ]
    block = mgr.build_memory_context_block(snippets, "t1")
    assert "<memory-context>" in block
    assert "tenant=t1" in block
    assert "Name: Alice" in block

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is my name?"},
    ]
    injected = MemoryManager.inject_memory_context_into_messages(messages, block)
    user_content = injected[-1]["content"]
    assert block in user_content
    assert "What is my name?" in user_content
    stripped = strip_memory_fences(user_content)
    assert "What is my name?" in stripped
    assert "<memory-context>" not in stripped


def test_notify_memory_tool_write_mirror():
    mirror = MirrorProvider()
    mgr = MemoryManager(external=mirror)
    mgr.notify_memory_tool_write(
        "t1",
        "u1",
        "add",
        {"target": "user", "content": "Lang: zh"},
    )
    mgr.flush_pending(timeout=2.0)
    assert len(mirror.writes) == 1
    assert mirror.writes[0][2] == "add"
    assert mirror.writes[0][3]["content"] == "Lang: zh"


@pytest.mark.asyncio
async def test_file_profile_provider_prefetch(tmp_path):
    profile_dir = tmp_path / "profiles" / "t1"
    profile_dir.mkdir(parents=True)
    profile = {
        "facts": [
            {"key": "company", "value": "Acme", "source": "crm"},
            {"key": "tier", "value": "gold", "source": "crm"},
        ]
    }
    with open(profile_dir / "u1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(profile, f, allow_unicode=True)

    from agent_platform.memory.adapters.file_external_memory_adapter import (
        FileExternalMemoryAdapter,
    )

    provider = FileProfileProvider(FileExternalMemoryAdapter(str(tmp_path / "profiles")))
    snippets = await provider.prefetch_turn("hello", _ctx())
    assert len(snippets) == 1
    assert "company: Acme" in snippets[0].content


@pytest.mark.asyncio
async def test_l1_store_triggers_memory_manager_mirror(tmp_path):
    mirror = MirrorProvider()
    mgr = MemoryManager(external=mirror)
    store = L1MemoryStore(
        store_dir=str(tmp_path / "mem"),
        memory_char_limit=200,
        user_char_limit=120,
        use_file_lock=False,
        on_memory_write=mgr.notify_memory_tool_write,
    )
    ctx = _ctx()
    result = store.add(ctx.tenant_id, ctx.user_id, "user", "Name: Bob")
    assert result["success"] is True
    mgr.flush_pending(timeout=2.0)
    assert any(w[2] == "add" for w in mirror.writes)


def test_provider_factory_none_by_default():
    assert resolve_l4_provider_name({"provider": "none"}) == "none"
    assert build_external_memory_provider({"provider": "none"}) is None


def test_provider_factory_file(tmp_path):
    cfg = {
        "provider": "file",
        "external_profiles_dir": str(tmp_path / "profiles"),
        "external_profiles_backend": "file",
    }
    provider = build_external_memory_provider(cfg)
    assert provider is not None
    assert provider.name == "file_profile"


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures():
    from agent_platform.memory.adapters.memory_manager import _CircuitBreaker

    mgr = MemoryManager(
        external=SlowProvider(delay_s=0.05),
        prefetch_timeout_ms=10,
        circuit_breaker=_CircuitBreaker(failure_threshold=2, reset_timeout_s=60),
    )
    await mgr.prefetch_all("a", _ctx())
    await mgr.prefetch_all("b", _ctx())
    assert mgr._circuit.is_open()
