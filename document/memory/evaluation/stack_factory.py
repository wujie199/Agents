# -*- coding: utf-8
"""Memory eval stack factory."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml

from core.composition.run_context import RunContext
from core.domain.context import RequestContext

from agent_platform.memory.adapters.config_loader import load_memory_config
from agent_platform.memory.adapters.external_factory import build_external_memory
from agent_platform.memory.adapters.hot_memory_compressor_adapter import (
    TruncatingHotMemoryCompressorAdapter,
)
from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
from agent_platform.memory.adapters.skill_memory_adapter import SkillMemoryAdapter
from agent_platform.memory.adapters.summarizer_adapter import TruncatingSummarizerAdapter
from agent_platform.infrastructure.privacy.adapter import PrivacyPortAdapter
from agent_platform.infrastructure.skills.adapter import SimpleSkillAdapter
from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)
from core.composition.memory_helpers import build_hot_memory


def _write_l4_profile(
    profiles_root: Path,
    tenant_id: str,
    user_id: str,
    facts: list[dict[str, str]],
) -> None:
    profile_dir = profiles_root / tenant_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "facts": [
            {"key": f["key"], "value": f["value"], "source": f.get("source", "eval")}
            for f in facts
        ]
    }
    (profile_dir / f"{user_id}.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8"
    )


async def build_eval_memory_stack(
    *,
    tmp_root: Path,
    enable_l4: bool = False,
    l4_facts: Optional[list[dict[str, str]]] = None,
    tenant_id: str = "eval_tenant",
    user_id: str = "eval_user",
    l1_backend: str = "file",
) -> tuple[MemoryPortAdapter, AsyncSQLiteRelationalAdapter, RunContext, Path]:
    cfg = load_memory_config()
    if l1_backend:
        cfg = {**cfg, "l1_store_backend": l1_backend}
    cfg = {**cfg, "skill_auto_extract_draft": False}
    archive_db = AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_root / "session_archive.db"), pool_size=2
    )
    store_dir = tmp_root / "memory"
    profiles_dir = tmp_root / "external_profiles"
    hot = build_hot_memory(cfg, archive_db=archive_db, store_dir_override=str(store_dir))

    external = None
    if enable_l4 and l4_facts:
        _write_l4_profile(profiles_dir, tenant_id, user_id, l4_facts)
        ext_cfg = {
            **cfg,
            "external_profiles_backend": "file",
            "external_profiles_dir": str(profiles_dir),
            "external_profile_cache_ttl": 0,
        }
        external = build_external_memory(ext_cfg)

    skills = SimpleSkillAdapter(skills_dir="skills/published")
    skill_memory = SkillMemoryAdapter(
        skills=skills,
        drafts_dir=str(tmp_root / "drafts"),
        meta_dir=str(tmp_root / "meta"),
    )
    memory = MemoryPortAdapter(
        store_dir=str(store_dir),
        archive_db=archive_db,
        hot_memory=hot,
        privacy=PrivacyPortAdapter(),
        skill_memory=skill_memory,
        summarizer=TruncatingSummarizerAdapter(max_chars=2000),
        compressor=TruncatingHotMemoryCompressorAdapter(),
        external_memory=external,
        external_merge_on_finalize=bool(enable_l4),
    )
    ctx = RunContext(
        request=RequestContext(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id="eval_session",
            trace_id="eval",
            channel="memory_eval",
        ),
        memory=memory,
    )
    ctx.extra = {}
    return memory, archive_db, ctx, store_dir


def try_load_models(config_dir: str = "config") -> Any:
    """可选加载 ModelRegistry；无 API key 时返回 None。"""
    try:
        from agent_platform.model.registry import ModelRegistry

        return ModelRegistry(config_path=f"{config_dir}/models.yml")
    except Exception:
        return None
