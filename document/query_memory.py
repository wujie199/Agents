#!/usr/bin/env python3
"""记忆子系统 CLI：L1 热记忆 + L2 冷档案 + L3 技能 + L4 外部画像。

交互验证 REPL: python document/memory_chat_repl.py --tenant tenant1 --user user1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta, TurnRecord
from core.composition.run_context import RunContext
from agent_platform.tools.adapters.tool_port_adapter import ToolPortAdapter
from agent_platform.infrastructure.privacy.adapter import PrivacyPortAdapter
from agent_platform.infrastructure.skills.adapter import SimpleSkillAdapter
from agent_platform.memory.adapters.config_loader import load_memory_config
from agent_platform.memory.adapters.archive_factory import build_archive_db
from agent_platform.memory.adapters.session_vector_factory import (
    build_session_vector_index,
)
from core.composition.production_factory import _build_memory_port, build_cache_port

log = logging.getLogger("query_memory")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _skill_run_context(ctx: RequestContext, memory) -> RunContext:
    from agent_platform.memory.memory_tool_registration import register_memory_tools

    tools = ToolPortAdapter(config_path=str(REPO_ROOT / "config" / "tools.yml"))
    register_memory_tools(tools, memory)
    return RunContext(request=ctx, memory=memory, tools=tools)


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _ctx(
    tenant_id: str,
    user_id: str,
    session_id: str,
    trace_id: str = "cli",
) -> RequestContext:
    return RequestContext(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        trace_id=trace_id,
        channel="cli",
    )


def _load_mem_cfg(config_dir: str) -> dict:
    config_path = (
        str(REPO_ROOT / config_dir)
        if not Path(config_dir).is_absolute()
        else config_dir
    )
    return load_memory_config(f"{config_path}/memory.yml"), config_path


def _build_memory(
    data_dir: Path,
    config_dir: str,
    db_name: str = "memory_cli.db",
    use_llm: bool = False,
    use_vector: bool = False,
    force_pg: bool = False,
    enable_cold_archive: Optional[bool] = None,
    with_rag_index: bool = False,
):
    mem_cfg, config_path = _load_mem_cfg(config_dir)
    data_path = str(data_dir)
    overrides: dict = {}
    if use_vector:
        overrides["enable_session_vector_index"] = True
    if force_pg:
        overrides["archive_backend"] = "postgresql"
    if enable_cold_archive is not None:
        overrides["enable_cold_archive"] = enable_cold_archive
    mem_cfg = {**mem_cfg, **overrides}

    archive = build_archive_db(
        mem_cfg,
        data_dir=data_path,
        db_name=db_name,
        force_backend="postgresql" if force_pg else None,
    )
    privacy = PrivacyPortAdapter()
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    cache = build_cache_port(prefix="sess")
    models = None
    if use_llm:
        from agent_platform.model.registry import ModelRegistry

        models = ModelRegistry(config_path=f"{config_path}/models.yml")

    index_port = None
    if with_rag_index:
        try:
            from agent_platform.storage.adapters.chroma.vector_adapter import (
                ChromaVectorAdapter,
            )
            from agent_platform.storage.adapters.graph.memory_graph_adapter import (
                MemoryGraphAdapter,
            )
            from core.composition.rag_factory_helpers import build_rag_stack

            vector_port = ChromaVectorAdapter(
                persist_directory=f"{data_path}/chroma_cli"
            )
            stack = build_rag_stack(
                models=models,
                vector_port=vector_port,
                cache_port=cache,
                config_dir=config_path,
                sql_port=archive,
                graph_port=MemoryGraphAdapter(),
                privacy_port=privacy,
                data_dir=data_path,
            )
            index_port = stack.index_port
        except Exception as exc:
            log.warning("RAG index stack unavailable for CLI: %s", exc)

    from agent_platform.storage.adapters.s3.s3_object_store_adapter import (
        S3ObjectStoreAdapter,
    )

    object_store = S3ObjectStoreAdapter(
        bucket_name=mem_cfg.get("cold_archive_bucket", "agents-storage"),
    )

    from agent_platform.infrastructure.secret.adapter import SecretPortAdapter

    memory, _memory_manager = _build_memory_port(
        config_dir=config_path,
        data_dir=data_path,
        archive_db=archive,
        privacy=privacy,
        skills=skills,
        cache=cache,
        models=models,
        store_dir_override=str(data_dir / "memory_cli_store"),
        object_store=object_store,
        index_port=index_port,
        secret=SecretPortAdapter(),
        mem_cfg_override=overrides,
    )
    return memory


def _memory_from_args(args: argparse.Namespace, **overrides):
    cold = True if getattr(args, "cold_archive", False) else None
    return _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
        enable_cold_archive=cold,
        **overrides,
    )


async def cmd_snapshot(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    snap = memory.compose_prompt_snapshot(
        _ctx(args.tenant, args.user, args.session or "cli")
    )
    print(f"hash={snap.hash} frozen={snap.frozen}\n")
    print(snap.memory_text)
    return 0


async def cmd_set_user(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, args.session or "cli")
    await memory.apply_memory_delta(
        ctx,
        MemoryDelta(key=args.key, value=args.value, source=args.source),
    )
    print(f"Applied delta: {args.key}={args.value} (source={args.source})")
    return 0


async def cmd_session_start(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, args.session)
    await memory.ensure_session(ctx)
    print(f"Session started: {args.session}")
    return 0


async def cmd_session_append(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, args.session)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role=args.role,
            content=args.content,
            ts=datetime.now().isoformat(),
            trace_id="cli",
        ),
    )
    print(f"Appended {args.role} message to session {args.session}")
    return 0


async def cmd_session_end(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, args.session)
    await memory.end_session(ctx, status=args.status)
    print(f"Session ended: {args.session} status={args.status}")
    return 0


async def cmd_session_list(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    rows = await memory.list_sessions(args.tenant, args.user, limit=args.limit)
    if not rows:
        print("（无会话）")
        return 0
    for row in rows:
        print(
            f"{row['session_id']}  status={row.get('status')}  "
            f"started={row.get('started_at')}  ended={row.get('ended_at')}"
        )
    return 0


async def cmd_session_turns(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, args.session)
    turns = await memory.list_turns(ctx, limit=args.limit, offset=args.offset)
    if not turns:
        print("（无消息）")
        return 0
    for t in turns:
        content = (t.get("content") or "")[:120]
        print(f"[{t.get('ts')}] {t.get('role')}: {content}")
    return 0


async def cmd_session_search(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=args.use_llm,
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, args.session)
    result = await memory.session_search(
        args.query, ctx, limit=args.limit, scope=args.scope
    )
    print(result)
    return 0


async def cmd_skill_list(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, "cli")
    rows = await memory.list_skills(ctx)
    if not rows:
        print("（无已发布技能）")
        return 0
    _print_json(rows)
    return 0


async def cmd_skill_get(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, "cli")
    detail = await memory.get_skill(args.skill_id, ctx)
    if detail is None:
        print(f"Skill not found: {args.skill_id}")
        return 1
    _print_json(detail)
    return 0


async def cmd_skill_run(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, args.session or "cli")
    inputs = json.loads(args.inputs) if args.inputs else {}
    run_ctx = _skill_run_context(ctx, memory)
    result = await memory.run_skill(args.skill_id, inputs, ctx, run_ctx)
    _print_json(
        {
            "skill_id": result.skill_id,
            "success": result.success,
            "steps_executed": result.steps_executed,
            "outputs": result.outputs,
            "error": result.error,
        }
    )
    return 0 if result.success else 1


async def cmd_skill_draft_list(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, "cli")
    rows = await memory.list_skill_drafts(ctx)
    if not rows:
        print("（无草稿）")
        return 0
    _print_json(rows)
    return 0


async def cmd_publish_skill(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, "cli")
    result = await memory.publish_skill(
        ctx, args.skill_id, remove_draft=not args.keep_draft
    )
    _print_json(result)
    return 0 if result.get("success") else 1


async def cmd_deprecate_skill(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, "cli")
    result = await memory.deprecate_skill(ctx, args.skill_id)
    _print_json(result)
    return 0 if result.get("success") else 1


async def cmd_activate_skill(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, "cli")
    result = await memory.activate_skill(ctx, args.skill_id)
    _print_json(result)
    return 0 if result.get("success") else 1


async def cmd_sync_skills(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    result = await memory.sync_skills_from(
        str(args.source_dir),
        remove_missing=args.remove_missing,
    )
    _print_json(result)
    return 0 if result.get("success") else 1


async def cmd_skill_runs_list(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, "cli")
    rows = await memory.list_skill_runs(
        ctx,
        skill_id=args.skill_id,
        limit=args.limit,
        offset=args.offset,
    )
    if not rows:
        print("（无 skill 执行记录）")
        return 0
    _print_json(rows)
    return 0


async def cmd_purge_tenant_l3(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    result = await memory.purge_tenant_l3(
        args.tenant,
        delete_runs=not args.keep_runs,
    )
    _print_json(result)
    return 0 if result.get("success") else 1


async def cmd_skill_search(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, "cli")
    hits = await memory.skill_search(args.query, ctx, limit=args.limit)
    if not hits:
        print("（无匹配技能）")
        return 0
    for i, s in enumerate(hits, 1):
        print(f"\n--- [{i}] {s.skill_id}: {s.title} ---")
        print(s.summary)
        if s.success_rate is not None:
            print(f"success_rate={s.success_rate}")
    return 0


async def cmd_confirm_pending(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, args.session or "cli")
    count = await memory.confirm_pending_deltas(ctx)
    print(f"Confirmed {count} pending delta(s)")
    return 0


async def cmd_resolve_entity(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, "cli")
    entity = await memory.resolve_entity(args.mention, ctx)
    if entity is None:
        print("（未解析到实体）")
        return 0
    print(f"mention={entity.mention} id={entity.canonical_id} name={entity.display_name}")
    return 0


async def cmd_finalize_session(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, args.session)
    await memory.finalize_session(ctx)
    snap = memory.compose_prompt_snapshot(ctx)
    print(f"Session finalized: {args.session}")
    print(f"snapshot hash={snap.hash}")
    print(snap.memory_text)
    return 0


async def cmd_profile_facts(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    facts = await memory.fetch_profile_facts(args.tenant, args.user)
    if not facts:
        print("（无外部画像 facts）")
        return 0
    _print_json(facts)
    return 0


async def cmd_profile_get(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    profile = await memory.get_profile(args.tenant, args.user)
    if not profile:
        print("（无外部画像文件）")
        return 0
    _print_json(profile)
    return 0


async def cmd_profile_set(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    facts = json.loads(args.facts)
    if not isinstance(facts, list):
        print("facts 必须是 JSON 数组")
        return 1
    count = await memory.set_profile_facts(args.tenant, args.user, facts)
    print(f"Upserted {count} fact(s)")
    return 0


async def cmd_profile_import(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    path = Path(args.file)
    if not path.is_file():
        print(f"File not found: {path}")
        return 1
    profile = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    await memory.import_profile(args.tenant, args.user, profile)
    print(f"Imported profile for {args.tenant}/{args.user}")
    return 0


async def cmd_list_profile_users(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    users = await memory.list_profile_users(args.tenant)
    if not users:
        print("（该租户无外部画像）")
        return 0
    for uid in users:
        print(uid)
    return 0


async def cmd_purge_tenant_l4(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    result = await memory.purge_tenant_l4(args.tenant)
    _print_json(result)
    return 0 if result.get("success") else 1


async def cmd_purge_user(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
        with_rag_index=not getattr(args, "no_rag", False),
    )
    summary = await memory.purge_user_data(args.tenant, args.user)
    print(f"Purged user data: tenant={args.tenant} user={args.user}")
    print(
        f"  messages_anonymized={summary.get('messages_anonymized', 0)} "
        f"cold_archives_deleted={summary.get('cold_archives_deleted', 0)} "
        f"rag_documents_deleted={summary.get('rag_documents_deleted', 0)}"
    )
    if summary.get("rag_doc_ids"):
        print(f"  rag_doc_ids={summary['rag_doc_ids']}")
    return 0


async def cmd_backfill_cold_search(args: argparse.Namespace) -> int:
    memory = _memory_from_args(args, enable_cold_archive=True)
    result = await memory.backfill_cold_search_index(
        tenant_id=args.tenant,
        user_id=getattr(args, "user", None),
        session_id=getattr(args, "session", None),
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(
        f"backfill cold search: indexed={result.get('indexed', 0)} "
        f"skipped={result.get('skipped', 0)} errors={result.get('errors', 0)} "
        f"dry_run={result.get('dry_run', False)}"
    )
    if result.get("reason"):
        print(f"  reason={result['reason']}")
    return 0 if result.get("errors", 0) == 0 else 1


async def cmd_backfill_all(args: argparse.Namespace) -> int:
    memory = _memory_from_args(
        args,
        enable_cold_archive=True,
        use_vector=True,
    )
    cold = await memory.backfill_cold_search_index(
        tenant_id=args.tenant,
        user_id=getattr(args, "user", None),
        session_id=getattr(args, "session", None),
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(
        f"[1/2] cold search: indexed={cold.get('indexed', 0)} "
        f"skipped={cold.get('skipped', 0)} errors={cold.get('errors', 0)}"
    )
    if cold.get("reason"):
        print(f"  reason={cold['reason']}")

    reindex: dict = {"indexed": 0, "errors": 0, "batches": 0, "reason": "dry_run"}
    if not args.dry_run:
        reindex = await memory.reindex_session_vectors(
            tenant_id=args.tenant,
            user_id=getattr(args, "user", None),
            session_id=getattr(args, "session", None),
            batch_size=args.batch_size,
        )
    print(
        f"[2/2] vector reindex: indexed={reindex.get('indexed', 0)} "
        f"errors={reindex.get('errors', 0)} batches={reindex.get('batches', 0)}"
    )
    if reindex.get("reason"):
        print(f"  reason={reindex['reason']}")

    errors = int(cold.get("errors", 0)) + int(reindex.get("errors", 0))
    return 0 if errors == 0 else 1


async def cmd_extract_skill_draft(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    ctx = _ctx(args.tenant, args.user, "cli")
    triggers = [t.strip() for t in args.triggers.split(",") if t.strip()]
    steps = json.loads(args.steps)
    draft_id = await memory.extract_skill_draft(
        ctx,
        args.title,
        triggers,
        steps,
        skill_id=args.skill_id or None,
    )
    print(f"Draft saved: {draft_id}")
    return 0


async def cmd_purge_expired(args: argparse.Namespace) -> int:
    memory = _memory_from_args(args)
    if memory.health().get("cold_archive") == "configured":
        result = await memory.archive_expired_sessions(retention_days=args.days)
        print(
            f"Cold archive: candidates={result.get('candidates', 0)} "
            f"archived={result.get('archived', 0)} "
            f"skipped={result.get('skipped', 0)} "
            f"errors={result.get('errors', 0)}"
        )
        return 0 if result.get("errors", 0) == 0 else 1
    count = await memory.purge_expired_sessions(retention_days=args.days)
    print(f"Purged expired sessions/messages: {count} rows affected")
    return 0


async def cmd_archive_expired(args: argparse.Namespace) -> int:
    memory = _memory_from_args(args, enable_cold_archive=True)
    result = await memory.archive_expired_sessions(retention_days=args.days)
    if result.get("reason") == "cold_archive_not_configured":
        print("Cold archive not configured (need object_store + enable_cold_archive)")
        return 1
    print(
        f"Archive expired: candidates={result.get('candidates', 0)} "
        f"archived={result.get('archived', 0)} "
        f"skipped={result.get('skipped', 0)} "
        f"errors={result.get('errors', 0)}"
    )
    return 0 if result.get("errors", 0) == 0 else 1


async def cmd_archive_session(args: argparse.Namespace) -> int:
    memory = _memory_from_args(args, enable_cold_archive=True)
    result = await memory.archive_session(args.session)
    if result.get("reason") == "cold_archive_not_configured":
        print("Cold archive not configured")
        return 1
    if result.get("skipped"):
        print(f"Session already archived: {args.session} key={result.get('object_key')}")
    else:
        print(
            f"Archived session={args.session} key={result.get('object_key')} "
            f"messages={result.get('message_count')} bytes={result.get('payload_bytes')}"
        )
    return 0


async def cmd_cold_list(args: argparse.Namespace) -> int:
    memory = _memory_from_args(args, enable_cold_archive=True)
    rows = await memory.list_cold_archives(args.tenant, args.user, limit=args.limit)
    if not rows:
        print("(no cold archives)")
        return 0
    for row in rows:
        print(
            f"{row.get('session_id')} archived_at={row.get('archived_at')} "
            f"messages={row.get('message_count')} key={row.get('object_key')}"
        )
    return 0


async def cmd_cold_fetch(args: argparse.Namespace) -> int:
    import json

    memory = _memory_from_args(args, enable_cold_archive=True)
    payload = await memory.fetch_cold_session(args.session)
    if payload is None:
        print(f"Cold session not found: {args.session}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


async def cmd_migrate_archive(args: argparse.Namespace) -> int:
    from agent_platform.memory.adapters.archive_migrate import (
        migrate_sqlite_to_postgresql,
    )
    from agent_platform.storage.adapters.sqlite.relational_adapter import (
        AsyncSQLiteRelationalAdapter,
    )

    mem_cfg, config_path = _load_mem_cfg(args.config_dir)

    sqlite_path = args.sqlite_path
    if sqlite_path is None:
        sqlite_path = mem_cfg.get("archive_sqlite_path") or str(
            args.data_dir / "memory_cli.db"
        )
    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        print(f"SQLite archive not found: {sqlite_path}")
        return 1

    sqlite_db = AsyncSQLiteRelationalAdapter(
        db_path=str(sqlite_path), pool_size=3
    )
    pg_db = build_archive_db(
        {**mem_cfg, "archive_backend": "postgresql"},
        data_dir=str(args.data_dir),
        force_backend="postgresql",
    )

    try:
        stats = await migrate_sqlite_to_postgresql(
            sqlite_db,
            pg_db,
            dry_run=args.dry_run,
            tenant_id=args.tenant,
            user_id=args.user,
        )
    finally:
        await sqlite_db.close()
        if hasattr(pg_db, "close"):
            await pg_db.close()

    if args.dry_run:
        print(
            f"Dry run: sessions={stats['sessions']} messages={stats['messages']} "
            f"tool_calls={stats['tool_calls']}"
        )
        return 0

    print(
        f"Migrated: sessions={stats['sessions_written']}/{stats['sessions']} "
        f"messages={stats['messages_written']}/{stats['messages']} "
        f"tool_calls={stats['tool_calls_written']}/{stats['tool_calls']} "
        f"errors={stats['errors']}"
    )

    if stats["errors"] > 0:
        return 1

    if args.reindex:
        if not args.tenant:
            print("--reindex requires --tenant (and optionally --user)")
            return 1
        mem_cfg_reindex = {
            **mem_cfg,
            "archive_backend": "postgresql",
            "enable_session_vector_index": True,
        }
        archive = build_archive_db(mem_cfg_reindex, data_dir=str(args.data_dir))
        session_vector_index = build_session_vector_index(
            mem_cfg_reindex,
            data_dir=str(args.data_dir),
            config_dir=config_path,
        )
        memory, _memory_manager = _build_memory_port(
            config_dir=config_path,
            data_dir=str(args.data_dir),
            archive_db=archive,
            privacy=PrivacyPortAdapter(),
            skills=SimpleSkillAdapter(skills_dir="skills/published"),
            cache=build_cache_port(prefix="sess"),
            session_vector_index=session_vector_index,
            session_hybrid_search=mem_cfg_reindex.get("session_hybrid_search", True),
        )
        result = await memory.reindex_session_vectors(
            tenant_id=args.tenant,
            user_id=args.user,
            session_id=args.session,
            batch_size=args.batch_size,
        )
        if result.get("reason"):
            print(f"Reindex skipped: {result['reason']}")
            return 1
        print(
            f"Reindex done: indexed={result['indexed']} errors={result['errors']} "
            f"batches={result['batches']}"
        )
        if result.get("errors", 0) > 0:
            return 1

    return 0


async def cmd_checkpoint_health(args: argparse.Namespace) -> int:
    mem_cfg, _ = _load_mem_cfg(args.config_dir)
    if getattr(args, "pg", False):
        mem_cfg = {**mem_cfg, "archive_backend": "postgresql"}

    archive = build_archive_db(
        mem_cfg,
        data_dir=str(args.data_dir),
        force_backend="postgresql" if getattr(args, "pg", False) else None,
    )
    from agent_platform.memory.adapters.relational_checkpointer_adapter import (
        RelationalCheckpointerAdapter,
    )

    try:
        if hasattr(archive, "_init_pool"):
            await archive._init_pool()
        cp = RelationalCheckpointerAdapter(archive)
        health = await cp.health_check()
        print(health)
        return 0 if health.get("status") == "healthy" else 1
    finally:
        if hasattr(archive, "close"):
            close = archive.close()
            if hasattr(close, "__await__"):
                await close


async def cmd_checkpoint_purge(args: argparse.Namespace) -> int:
    mem_cfg, _ = _load_mem_cfg(args.config_dir)
    archive = build_archive_db(mem_cfg, data_dir=str(args.data_dir))
    from agent_platform.memory.adapters.relational_checkpointer_adapter import (
        RelationalCheckpointerAdapter,
    )

    days = args.days or mem_cfg.get(
        "checkpoint_retention_days", mem_cfg.get("retention_days", 90)
    )
    try:
        if hasattr(archive, "_init_pool"):
            await archive._init_pool()
        cp = RelationalCheckpointerAdapter(archive)
        count = await cp.purge_older_than(int(days))
        print(f"purged checkpoints: {count} (retention_days={days})")
        return 0
    finally:
        if hasattr(archive, "close"):
            close = archive.close()
            if hasattr(close, "__await__"):
                await close


async def cmd_checkpoint_list(args: argparse.Namespace) -> int:
    mem_cfg, _ = _load_mem_cfg(args.config_dir)
    archive = build_archive_db(mem_cfg, data_dir=str(args.data_dir))
    from agent_platform.memory.adapters.relational_checkpointer_adapter import (
        RelationalCheckpointerAdapter,
    )

    try:
        if hasattr(archive, "_init_pool"):
            await archive._init_pool()
        cp = RelationalCheckpointerAdapter(archive)
        rows = await cp.list_threads(args.tenant, limit=args.limit)
        for row in rows:
            print(row)
        return 0
    finally:
        if hasattr(archive, "close"):
            close = archive.close()
            if hasattr(close, "__await__"):
                await close


async def cmd_archive_health(args: argparse.Namespace) -> int:
    mem_cfg, config_path = _load_mem_cfg(args.config_dir)
    if getattr(args, "pg", False):
        mem_cfg = {**mem_cfg, "archive_backend": "postgresql"}

    archive = build_archive_db(
        mem_cfg,
        data_dir=str(args.data_dir),
        force_backend="postgresql" if getattr(args, "pg", False) else None,
    )

    try:
        if hasattr(archive, "_init_pool"):
            await archive._init_pool()
        if hasattr(archive, "health_check"):
            health = await archive.health_check()
        else:
            health = await archive.health()

        backend = mem_cfg.get("archive_backend", "sqlite")
        print(f"archive_backend={backend}")
        print(f"archive_db={health}")

        counts = {}
        if backend == "postgresql":
            async with archive._get_connection() as conn:
                for table in ("sessions", "messages", "tool_calls"):
                    counts[table] = await conn.fetchval(
                        f"SELECT COUNT(*) FROM {table}"
                    )
        else:
            for table in ("sessions", "messages", "tool_calls"):
                rows = await archive.execute(f"SELECT COUNT(*) AS c FROM {table}")
                counts[table] = dict(rows[0])["c"] if rows else 0
        print(f"row_counts={counts}")

        memory = _build_memory(
            args.data_dir,
            args.config_dir,
            force_pg=getattr(args, "pg", False),
            use_vector=getattr(args, "vector", False),
        )
        print(f"memory_port={memory.health()}")

        return 0 if health.get("status") == "healthy" else 1
    finally:
        if hasattr(archive, "close"):
            await archive.close()


async def cmd_reindex(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=True,
        force_pg=getattr(args, "pg", False),
    )
    result = await memory.reindex_session_vectors(
        tenant_id=args.tenant,
        user_id=args.user,
        session_id=args.session,
        batch_size=args.batch_size,
    )
    if result.get("reason"):
        print(f"Reindex skipped: {result['reason']}")
        return 1
    print(
        f"Reindex done: indexed={result['indexed']} errors={result['errors']} "
        f"batches={result['batches']}"
    )
    return 0 if result.get("errors", 0) == 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="记忆子系统 CLI")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--config-dir", default="config")
    p.add_argument("--use-llm", action="store_true", help="启用 LLM 摘要/压缩")
    p.add_argument("--vector", action="store_true", help="启用 L2 向量混合检索")
    p.add_argument(
        "--pg",
        action="store_true",
        help="L2 归档强制使用 PostgreSQL（等同 archive_backend: postgresql）",
    )
    p.add_argument(
        "--cold-archive",
        action="store_true",
        help="启用 L2 冷归档（过期会话导出对象存储后删在线明细）",
    )

    sub = p.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="L1 热记忆快照")
    snap.add_argument("--tenant", required=True)
    snap.add_argument("--user", required=True)
    snap.add_argument("--session", default="cli")
    snap.set_defaults(func=cmd_snapshot)

    su = sub.add_parser("set-user", help="L1 写入用户偏好")
    su.add_argument("--tenant", required=True)
    su.add_argument("--user", required=True)
    su.add_argument("--session", default="cli")
    su.add_argument("--key", required=True)
    su.add_argument("--value", required=True)
    su.add_argument("--source", default="user", choices=["user", "memory"])
    su.set_defaults(func=cmd_set_user)

    ss = sub.add_parser("session-start", help="L2 创建会话")
    ss.add_argument("--tenant", required=True)
    ss.add_argument("--user", required=True)
    ss.add_argument("--session", required=True)
    ss.set_defaults(func=cmd_session_start)

    sa = sub.add_parser("session-append", help="L2 追加消息")
    sa.add_argument("--tenant", required=True)
    sa.add_argument("--user", required=True)
    sa.add_argument("--session", required=True)
    sa.add_argument("--role", required=True, choices=["user", "assistant", "system"])
    sa.add_argument("--content", required=True)
    sa.set_defaults(func=cmd_session_append)

    se = sub.add_parser("session-end", help="L2 结束会话")
    se.add_argument("--tenant", required=True)
    se.add_argument("--user", required=True)
    se.add_argument("--session", required=True)
    se.add_argument("--status", default="closed")
    se.set_defaults(func=cmd_session_end)

    sl = sub.add_parser("session-list", help="L2 列出会话")
    sl.add_argument("--tenant", required=True)
    sl.add_argument("--user", required=True)
    sl.add_argument("--limit", type=int, default=20)
    sl.set_defaults(func=cmd_session_list)

    st = sub.add_parser("session-turns", help="L2 列出会话消息")
    st.add_argument("--tenant", required=True)
    st.add_argument("--user", required=True)
    st.add_argument("--session", required=True)
    st.add_argument("--limit", type=int, default=50)
    st.add_argument("--offset", type=int, default=0)
    st.set_defaults(func=cmd_session_turns)

    sr = sub.add_parser("session-search", help="L2 检索会话")
    sr.add_argument("--tenant", required=True)
    sr.add_argument("--user", required=True)
    sr.add_argument("--session", required=True)
    sr.add_argument("--query", required=True)
    sr.add_argument("--limit", type=int, default=5)
    sr.add_argument(
        "--scope",
        default="session",
        choices=["session", "user"],
        help="session=仅当前会话; user=该用户全部会话",
    )
    sr.set_defaults(func=cmd_session_search)

    sk = sub.add_parser("skill-search", help="L3 技能搜索")
    sk.add_argument("--tenant", default="default")
    sk.add_argument("--user", default="cli")
    sk.add_argument("--query", required=True)
    sk.add_argument("--limit", type=int, default=3)
    sk.set_defaults(func=cmd_skill_search)

    skl = sub.add_parser("skill-list", help="L3 列出已发布技能")
    skl.add_argument("--tenant", default="default")
    skl.add_argument("--user", default="cli")
    skl.set_defaults(func=cmd_skill_list)

    skg = sub.add_parser("skill-get", help="L3 查看技能详情")
    skg.add_argument("--tenant", default="default")
    skg.add_argument("--user", default="cli")
    skg.add_argument("--skill-id", required=True)
    skg.set_defaults(func=cmd_skill_get)

    skr = sub.add_parser("skill-run", help="L3 执行技能")
    skr.add_argument("--tenant", default="default")
    skr.add_argument("--user", default="cli")
    skr.add_argument("--session", default="cli")
    skr.add_argument("--skill-id", required=True)
    skr.add_argument(
        "--inputs",
        default="",
        help='JSON 输入，如 \'{"message":"hello"}\'',
    )
    skr.set_defaults(func=cmd_skill_run)

    skd = sub.add_parser("skill-draft-list", help="L3 列出草稿")
    skd.add_argument("--tenant", required=True)
    skd.add_argument("--user", default="cli")
    skd.set_defaults(func=cmd_skill_draft_list)

    psk = sub.add_parser("publish-skill", help="L3 发布草稿到 skills/published")
    psk.add_argument("--tenant", required=True)
    psk.add_argument("--user", default="cli")
    psk.add_argument("--skill-id", required=True)
    psk.add_argument(
        "--keep-draft",
        action="store_true",
        help="发布后保留草稿目录",
    )
    psk.set_defaults(func=cmd_publish_skill)

    dsk = sub.add_parser("deprecate-skill", help="L3 标记技能为 deprecated")
    dsk.add_argument("--tenant", default="default")
    dsk.add_argument("--user", default="cli")
    dsk.add_argument("--skill-id", required=True)
    dsk.set_defaults(func=cmd_deprecate_skill)

    ask = sub.add_parser("activate-skill", help="L3 恢复技能为 active")
    ask.add_argument("--tenant", default="default")
    ask.add_argument("--user", default="cli")
    ask.add_argument("--skill-id", required=True)
    ask.set_defaults(func=cmd_activate_skill)

    ssy = sub.add_parser(
        "sync-skills",
        help="从源目录同步 skill.yaml 到 skills/published",
    )
    ssy.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "skills" / "source",
        help="源目录（每个子目录含 skill.yaml）",
    )
    ssy.add_argument(
        "--remove-missing",
        action="store_true",
        help="删除 published 中源目录不存在的技能",
    )
    ssy.set_defaults(func=cmd_sync_skills)

    srl = sub.add_parser("skill-runs-list", help="L3 列出 skill 执行审计")
    srl.add_argument("--tenant", required=True)
    srl.add_argument("--user", required=True)
    srl.add_argument("--skill-id", default=None)
    srl.add_argument("--limit", type=int, default=20)
    srl.add_argument("--offset", type=int, default=0)
    srl.set_defaults(func=cmd_skill_runs_list)

    ptl = sub.add_parser(
        "purge-tenant-l3",
        help="L3 清理租户 drafts + meta（可选 skill_runs）",
    )
    ptl.add_argument("--tenant", required=True)
    ptl.add_argument(
        "--keep-runs",
        action="store_true",
        help="保留 skill_runs 审计记录",
    )
    ptl.set_defaults(func=cmd_purge_tenant_l3)

    cp = sub.add_parser("confirm-pending", help="L1 确认 HITL pending 写入")
    cp.add_argument("--tenant", required=True)
    cp.add_argument("--user", required=True)
    cp.add_argument("--session", default="cli")
    cp.set_defaults(func=cmd_confirm_pending)

    re = sub.add_parser("resolve-entity", help="L4 实体解析")
    re.add_argument("--tenant", required=True)
    re.add_argument("--user", required=True)
    re.add_argument("--mention", required=True)
    re.set_defaults(func=cmd_resolve_entity)

    fs = sub.add_parser("finalize-session", help="L1+L4 会话结束合并 pending 与外部画像")
    fs.add_argument("--tenant", required=True)
    fs.add_argument("--user", required=True)
    fs.add_argument("--session", required=True)
    fs.set_defaults(func=cmd_finalize_session)

    pf = sub.add_parser("profile-facts", help="L4 列出外部画像 facts")
    pf.add_argument("--tenant", required=True)
    pf.add_argument("--user", required=True)
    pf.set_defaults(func=cmd_profile_facts)

    pg = sub.add_parser("profile-get", help="L4 读取完整外部画像 YAML")
    pg.add_argument("--tenant", required=True)
    pg.add_argument("--user", required=True)
    pg.set_defaults(func=cmd_profile_get)

    ps = sub.add_parser("profile-set", help="L4 upsert facts 到外部画像")
    ps.add_argument("--tenant", required=True)
    ps.add_argument("--user", required=True)
    ps.add_argument(
        "--facts",
        required=True,
        help='JSON 数组，如 \'[{"key":"部门","value":"研发","source":"ldap"}]\'',
    )
    ps.set_defaults(func=cmd_profile_set)

    pi = sub.add_parser("profile-import", help="L4 从 YAML 文件导入外部画像")
    pi.add_argument("--tenant", required=True)
    pi.add_argument("--user", required=True)
    pi.add_argument("--file", required=True, type=Path)
    pi.set_defaults(func=cmd_profile_import)

    lpu = sub.add_parser("list-profile-users", help="L4 列出租户下有画像的用户")
    lpu.add_argument("--tenant", required=True)
    lpu.set_defaults(func=cmd_list_profile_users)

    ptl4 = sub.add_parser(
        "purge-tenant-l4",
        help="L4 清理租户全部外部画像 YAML",
    )
    ptl4.add_argument("--tenant", required=True)
    ptl4.set_defaults(func=cmd_purge_tenant_l4)

    pu = sub.add_parser("purge-user", help="合规：擦除用户 L1+L2+RAG+L3+L4")
    pu.add_argument("--tenant", required=True)
    pu.add_argument("--user", required=True)
    pu.add_argument(
        "--no-rag",
        action="store_true",
        help="不联动删除 RAG 文档（默认会删 metadata 含 user_id 的文档）",
    )
    pu.set_defaults(func=cmd_purge_user)

    pe = sub.add_parser("purge-expired", help="L2 清理过期会话")
    pe.add_argument("--days", type=int, default=90)
    pe.set_defaults(func=cmd_purge_expired)

    ae = sub.add_parser(
        "archive-expired",
        help="L2 冷归档：导出过期会话到对象存储并删除在线明细",
    )
    ae.add_argument("--days", type=int, default=90)
    ae.set_defaults(func=cmd_archive_expired)

    ars = sub.add_parser(
        "archive-session",
        help="L2 冷归档：立即归档指定会话（不要求过期）",
    )
    ars.add_argument("--tenant", required=True)
    ars.add_argument("--user", required=True)
    ars.add_argument("--session", required=True)
    ars.set_defaults(func=cmd_archive_session)

    cl = sub.add_parser("cold-list", help="列出用户冷归档索引")
    cl.add_argument("--tenant", required=True)
    cl.add_argument("--user", required=True)
    cl.add_argument("--limit", type=int, default=20)
    cl.set_defaults(func=cmd_cold_list)

    cf = sub.add_parser("cold-fetch", help="从对象存储拉取冷归档会话 JSON")
    cf.add_argument("--tenant", required=True)
    cf.add_argument("--user", required=True)
    cf.add_argument("--session", required=True)
    cf.set_defaults(func=cmd_cold_fetch)

    ri = sub.add_parser(
        "reindex",
        help="将历史 messages 批量写入向量索引（需 --vector / enable_session_vector_index）",
    )
    ri.add_argument("--tenant", required=True)
    ri.add_argument("--user", default=None, help="不填则重建该 tenant 下全部用户消息")
    ri.add_argument("--session", default=None, help="不填则不限会话")
    ri.add_argument("--batch-size", type=int, default=200)
    ri.set_defaults(func=cmd_reindex)

    bf = sub.add_parser(
        "backfill-cold-search",
        help="为历史冷归档补建 DB 检索索引（需 --cold-archive）",
    )
    bf.add_argument("--tenant", required=True)
    bf.add_argument("--user", default=None)
    bf.add_argument("--session", default=None)
    bf.add_argument("--limit", type=int, default=100, help="每批处理的冷归档会话数")
    bf.add_argument("--force", action="store_true", help="强制重建已有索引")
    bf.add_argument("--dry-run", action="store_true")
    bf.set_defaults(func=cmd_backfill_cold_search)

    bfa = sub.add_parser(
        "backfill-all",
        help="历史数据：冷归档检索索引 + 会话向量 reindex（需 --cold-archive --vector）",
    )
    bfa.add_argument("--tenant", required=True)
    bfa.add_argument("--user", default=None)
    bfa.add_argument("--session", default=None)
    bfa.add_argument("--limit", type=int, default=100)
    bfa.add_argument("--batch-size", type=int, default=200)
    bfa.add_argument("--force", action="store_true")
    bfa.add_argument("--dry-run", action="store_true")
    bfa.set_defaults(func=cmd_backfill_all)

    esd = sub.add_parser("extract-skill-draft", help="L3 从参数创建技能草稿")
    esd.add_argument("--tenant", required=True)
    esd.add_argument("--user", default="cli")
    esd.add_argument("--title", required=True)
    esd.add_argument("--triggers", required=True, help="逗号分隔 trigger 列表")
    esd.add_argument(
        "--steps",
        required=True,
        help='JSON steps 数组，如 \'[{"action":"echo","tool":"skill_echo"}]\'',
    )
    esd.add_argument("--skill-id", default=None)
    esd.set_defaults(func=cmd_extract_skill_draft)

    mg = sub.add_parser(
        "migrate-archive",
        help="SQLite L2 归档迁移到 PostgreSQL（见 document/memory/MIGRATE_ARCHIVE.md）",
    )
    mg.add_argument(
        "--sqlite-path",
        type=Path,
        default=None,
        help="源 SQLite 路径，默认 config archive_sqlite_path 或 data/memory_cli.db",
    )
    mg.add_argument("--dry-run", action="store_true", help="只统计，不写入 PG")
    mg.add_argument("--tenant", default=None, help="只迁移该租户")
    mg.add_argument("--user", default=None, help="只迁移该用户（需配合 --tenant）")
    mg.add_argument(
        "--reindex",
        action="store_true",
        help="迁移成功后对 PG 数据执行向量 reindex（需 --tenant）",
    )
    mg.add_argument("--batch-size", type=int, default=200, help="reindex 批大小")
    mg.set_defaults(func=cmd_migrate_archive)

    ch = sub.add_parser("checkpoint-health", help="Checkpointer 表健康检查")
    ch.set_defaults(func=cmd_checkpoint_health)

    cp = sub.add_parser("checkpoint-purge", help="清理超期 graph_checkpoints")
    cp.add_argument(
        "--days",
        type=int,
        default=None,
        help="保留天数（默认 checkpoint_retention_days）",
    )
    cp.set_defaults(func=cmd_checkpoint_purge)

    cl = sub.add_parser("checkpoint-list", help="列出最近 checkpoint 线程")
    cl.add_argument("--tenant", required=True)
    cl.add_argument("--limit", type=int, default=20)
    cl.set_defaults(func=cmd_checkpoint_list)

    ah = sub.add_parser(
        "archive-health",
        help="检查 L2 归档库连接与表（PG 生产冒烟）",
    )
    ah.set_defaults(func=cmd_archive_health)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
