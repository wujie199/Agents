#!/usr/bin/env python3
"""记忆子系统 CLI：L1 热记忆 + L2 冷档案 + L3 技能搜索（不依赖 Agent）。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta, TurnRecord
from agent_platform.infrastructure.privacy.adapter import PrivacyPortAdapter
from agent_platform.infrastructure.skills.adapter import SimpleSkillAdapter
from agent_platform.storage.adapters.memory.async_cache_adapter import (
    AsyncMemoryCacheAdapter,
)
from agent_platform.memory.adapters.config_loader import load_memory_config
from agent_platform.memory.adapters.archive_factory import build_archive_db
from agent_platform.memory.adapters.session_vector_factory import (
    build_session_vector_index,
)
from core.composition.production_factory import _build_memory_port

log = logging.getLogger("query_memory")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


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
):
    mem_cfg, config_path = _load_mem_cfg(config_dir)
    data_path = str(data_dir)
    if use_vector:
        mem_cfg = {**mem_cfg, "enable_session_vector_index": True}
    if force_pg:
        mem_cfg = {**mem_cfg, "archive_backend": "postgresql"}
    if enable_cold_archive is not None:
        mem_cfg = {**mem_cfg, "enable_cold_archive": enable_cold_archive}

    archive = build_archive_db(
        mem_cfg,
        data_dir=data_path,
        db_name=db_name,
        force_backend="postgresql" if force_pg else None,
    )
    privacy = PrivacyPortAdapter()
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    cache = AsyncMemoryCacheAdapter(prefix="sess")
    models = None
    if use_llm:
        from agent_platform.model.registry import ModelRegistry

        models = ModelRegistry(config_path=f"{config_path}/models.yml")

    session_vector_index = build_session_vector_index(
        mem_cfg, data_dir=data_path, config_dir=config_path
    )

    from agent_platform.storage.adapters.s3.s3_object_store_adapter import (
        S3ObjectStoreAdapter,
    )

    object_store = S3ObjectStoreAdapter(
        bucket_name=mem_cfg.get("cold_archive_bucket", "agents-storage"),
    )

    return _build_memory_port(
        config_dir=config_path,
        data_dir=data_path,
        archive_db=archive,
        privacy=privacy,
        skills=skills,
        cache=cache,
        models=models,
        store_dir_override=str(data_dir / "memory_cli_store"),
        session_vector_index=session_vector_index,
        session_hybrid_search=mem_cfg.get("session_hybrid_search", True),
        object_store=object_store,
        enable_cold_archive=mem_cfg.get("enable_cold_archive", False),
        cold_archive_prefix=mem_cfg.get("cold_archive_prefix", "l2/cold"),
        cold_archive_compress=mem_cfg.get("cold_archive_compress", True),
    )


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


async def cmd_purge_user(args: argparse.Namespace) -> int:
    memory = _build_memory(
        args.data_dir,
        args.config_dir,
        use_llm=getattr(args, "use_llm", False),
        use_vector=getattr(args, "vector", False),
        force_pg=getattr(args, "pg", False),
    )
    await memory.purge_user_data(args.tenant, args.user)
    print(f"Purged user data: tenant={args.tenant} user={args.user}")
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
        memory = _build_memory_port(
            config_dir=config_path,
            data_dir=str(args.data_dir),
            archive_db=archive,
            privacy=PrivacyPortAdapter(),
            skills=SimpleSkillAdapter(skills_dir="skills/published"),
            cache=AsyncMemoryCacheAdapter(prefix="sess"),
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

    pu = sub.add_parser("purge-user", help="合规：擦除用户 L1+L2")
    pu.add_argument("--tenant", required=True)
    pu.add_argument("--user", required=True)
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
