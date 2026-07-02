"""MemoryProvider 工厂：从 memory.yml 构建外部 Provider。"""

from __future__ import annotations

from typing import Any, Optional

from agent_platform.memory.adapters.external_factory import build_external_memory
from agent_platform.memory.adapters.memory_providers.file_profile_provider import (
    FileProfileProvider,
)
from agent_platform.memory.adapters.noop_external_adapter import (
    NoOpExternalMemoryAdapter,
)
from core.ports.memory.provider import MemoryProvider


def resolve_l4_provider_name(cfg: dict[str, Any]) -> str:
    """解析 l4 provider 名称（展平后键为 provider）。"""
    raw = cfg.get("provider") or cfg.get("l4_provider") or "none"
    return str(raw).strip().lower()


def build_external_memory_provider(
    cfg: dict[str, Any],
    *,
    external_memory: Any = None,
) -> Optional[MemoryProvider]:
    """按配置构建至多一个外部 MemoryProvider。"""
    name = resolve_l4_provider_name(cfg)
    if name in ("none", "noop", "builtin_only", "builtin", ""):
        return None

    backend = str(cfg.get("external_profiles_backend", "file")).lower()
    if name in ("file", "http", "external"):
        ext = external_memory if external_memory is not None else build_external_memory(cfg)
        if isinstance(ext, NoOpExternalMemoryAdapter) and name != "external":
            return None
        max_chars = int(cfg.get("l4_prefetch_max_chars", 2000))
        return FileProfileProvider(ext, max_prefetch_chars=max_chars)

    if backend in ("file", "http"):
        ext = external_memory if external_memory is not None else build_external_memory(cfg)
        if isinstance(ext, NoOpExternalMemoryAdapter):
            return None
        max_chars = int(cfg.get("l4_prefetch_max_chars", 2000))
        return FileProfileProvider(ext, max_prefetch_chars=max_chars)

    return None
