#!/usr/bin/env python3
"""记忆 + RAG 对话 REPL。

用法:
  cd /path/to/Agents
  python app/chat_repl.py --tenant tenant1 --user user1 --session chat1
  python app/chat_repl.py --no-rag
  python app/chat_repl.py --engine direct   # 不用 LangGraph
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.domain.context import RequestContext
from app.agents.context_factory import build_chat_run_context

from app.agents.chat_config import load_chat_config
from app.agents.chat_service import (
    ChatSessionHandle,
    execute_chat_turn,
    stream_chat_turn_events,
)
from app.agents.chat_langgraph import create_chat_langgraph_session_async
from app.agents.memory_metrics import get_memory_metric_stats, prometheus_text
from app.agents.memory_views import list_pending_l1_deltas
from app.agents.enterprise_memory import (
    confirm_pending_l1,
    get_memory_status,
    list_user_sessions,
)
from app.agents.react_loop import end_agent_session
from app.agents.debug_trace import agent_debug, agent_debug_log_path, set_debug_console, set_debug_quiet
from app.agents.memory_runtime_debug import (
    collect_memory_runtime_status,
    debug_log_path,
    format_layer_triggers,
    format_memory_runtime_detailed,
    format_memory_runtime_summary,
    format_post_turn_debug,
    is_memory_runtime_debug,
    log_memory_runtime_status,
    log_turn_trace,
    memory_debug_console_enabled,
    resolve_memory_trace,
    set_memory_runtime_debug,
    set_memory_runtime_verbose,
)

HELP = """
命令:
  直接输入文字              一轮对话（L1 + 可选 RAG + L2）
  /snapshot                 查看 L1 热记忆快照
  /turns                    列出本会话 L2 消息
  /confirm                  确认 pending L1 记忆（HITL）
  /pending                  查看待确认 L1 记忆
  /sessions                 列出本会话用户的历史 session（L2）
  /status                   记忆状态摘要（L1/L2/配置）
  /debug-memory             完整 L1-L4/RAG 调试详情（控制台 + NDJSON）
  /triggers                 本轮 L1-L4/RAG 触发链（何时 ON/SKIP）
  /cache-test [关键词]      同一 query 连调两次 session_search，验证 Redis 缓存 HIT
  /metrics                  记忆子系统指标（ObservabilityPort）
  /new [session_id]         切换新会话（清空 L2 上下文，可选自定义 id）
  /help                     本帮助
  /quit, /q, exit           结束会话（L4→L1 finalize）

启动参数:
  --no-rag      关闭知识库检索
  --engine      langgraph（默认）| direct
  --debug       详细调试：L1-L4/RAG/Context 全链路 NDJSON 日志
  --debug-quiet  同 --debug，但不打印控制台 debug 摘要（仅写日志文件）
  --no-tools    关闭 Path B 记忆工具（L1/L2/L3/L4 全套）
  --profile     dev（默认）| production
  --stream      流式打印 assistant 回复
"""


async def _handle_line(
    line: str,
    *,
    run_ctx,
    enable_rag: bool,
    chat_cfg,
    engine: str,
    session_handle: ChatSessionHandle,
    stream_output: bool = False,
    debug: bool = False,
) -> bool:
    if line in ("/quit", "/q", "exit"):
        return False

    if line in ("/help", "?"):
        print(HELP)
        return True

    memory = run_ctx.require_memory()
    req = run_ctx.request

    if line == "/snapshot":
        snap = memory.compose_prompt_snapshot(req)
        agent_debug(
            "MEM-L1",
            "chat_repl:/snapshot",
            "手动 L1 快照",
            {
                "hash": snap.hash,
                "memory_chars": len(snap.memory_text or ""),
                "memory_text": snap.memory_text,
            },
        )
        print(f"\n--- L1 hash={snap.hash} ---\n{snap.memory_text}\n")
        return True

    if line == "/turns":
        rows = await memory.list_turns(req, limit=50)
        agent_debug(
            "MEM-L2-HIST",
            "chat_repl:/turns",
            "手动 L2 列表",
            {
                "count": len(rows),
                "rows": [
                    {
                        "role": r.get("role"),
                        "ts": r.get("ts"),
                        "chars": len(r.get("content") or ""),
                        "preview": (r.get("content") or "")[:100],
                    }
                    for r in rows
                ],
            },
        )
        if not rows:
            print("（本会话暂无消息）")
            return True
        for row in rows:
            content = (row.get("content") or "")[:200]
            print(f"  [{row.get('ts', '')}] {row.get('role')}: {content}")
        return True

    if line == "/confirm":
        n = await confirm_pending_l1(run_ctx)
        print(f"\n[ok] 已确认 {n} 条 pending L1 记忆\n")
        return True

    if line == "/pending":
        pending = list_pending_l1_deltas(run_ctx)
        if not pending:
            print("\n（无 pending L1 记忆）\n")
            return True
        print("\n--- pending L1 ---")
        for row in pending:
            print(f"  {row.get('key')} = {row.get('value')}  [{row.get('source')}]")
        print()
        return True

    if line == "/sessions":
        rows = await list_user_sessions(run_ctx, limit=20)
        if not rows:
            print("\n（无历史 session）\n")
            return True
        print("\n--- sessions ---")
        for row in rows:
            print(
                f"  {row.get('session_id')}  status={row.get('status')}  "
                f"started={row.get('started_at', '')}"
            )
        print()
        return True

    if line == "/status":
        status = await get_memory_status(run_ctx)
        print(f"\n--- memory status ---")
        print(f"  L1 hash={status['l1_hash']}  chars={status['l1_chars']}")
        print(f"  pending L1={status['pending_l1_count']}")
        cfg = status.get("config") or {}
        print(
            f"  archive={cfg.get('archive_backend')}  "
            f"l1={cfg.get('l1_store_backend')}  "
            f"cold={cfg.get('enable_cold_archive')}  "
            f"vector={cfg.get('enable_session_vector_index')}"
        )
        metrics = status.get("metrics") or {}
        if metrics:
            print("  metrics:")
            for name, stats in metrics.items():
                print(f"    {name}: count={stats.get('count')} sum={stats.get('sum')}")
        if is_memory_runtime_debug():
            rt = await log_memory_runtime_status(
                run_ctx, event="repl_status", console=False
            )
            print(format_memory_runtime_detailed(rt))
        else:
            print()
        return True

    if line == "/triggers":
        print("\n" + format_layer_triggers(run_ctx) + "\n")
        return True

    if line == "/debug-memory":
        rt = await log_memory_runtime_status(
            run_ctx, event="repl_debug_memory", console=False, force=True
        )
        print(format_memory_runtime_detailed(rt))
        print(f"  → NDJSON: {debug_log_path()}")
        print(f"  → trace:  {agent_debug_log_path()}\n")
        return True

    if line.startswith("/cache-test"):
        parts = line.split(maxsplit=1)
        probe_query = parts[1].strip() if len(parts) > 1 else "Phoenix"
        cache = (run_ctx.extra or {}).get("cache")
        adapter = type(cache).__name__ if cache else "none"
        print(f"\n--- session_search cache probe ---")
        print(f"  cache_adapter={adapter}")
        print(f"  query={probe_query!r}  scope=session  limit=5")
        print("  (连续调用两次 session_search，第 2 次应 cache HIT)\n")
        r1 = await memory.session_search(
            probe_query, req, limit=5, scope="session"
        )
        r2 = await memory.session_search(
            probe_query, req, limit=5, scope="session"
        )
        print(f"  1st: len={len(r1)}  preview={r1[:120]!r}")
        print(f"  2nd: len={len(r2)}  preview={r2[:120]!r}")
        print(f"  identical={r1 == r2}")
        print(f"  → 日志: {debug_log_path()} (看 cache_hit: false 然后 true)\n")
        return True

    if line == "/metrics":
        stats = get_memory_metric_stats(run_ctx)
        if not stats:
            print("\n（暂无 memory.* 指标）\n")
            return True
        print("\n--- memory metrics ---")
        for name, row in stats.items():
            print(
                f"  {name}: count={row.get('count')} sum={row.get('sum')} "
                f"avg={row.get('avg'):.2f}"
            )
        prom = prometheus_text(run_ctx).strip()
        if prom and prom != "# no metrics":
            print("\n--- prometheus ---")
            print(prom)
        print()
        return True

    if line == "/new" or line.startswith("/new "):
        import uuid
        from dataclasses import replace

        from app.agents.chat_langgraph import create_chat_langgraph_session_async

        parts = line.split(maxsplit=1)
        new_sid = (
            parts[1].strip()
            if len(parts) > 1 and parts[1].strip()
            else f"chat-{uuid.uuid4().hex[:8]}"
        )
        new_req = replace(req, session_id=new_sid)
        session_handle.run_ctx = replace(run_ctx, request=new_req)
        session_handle.lg_session = None
        if engine == "langgraph":
            from app.agents.chat_langgraph import create_chat_langgraph_session_async

            session_handle.lg_session = await create_chat_langgraph_session_async(
                session_handle.run_ctx, chat_cfg=chat_cfg
            )
        await memory.ensure_session(new_req)
        print(f"\n→ 新会话 session_id={new_sid}（L2 历史独立，L1 仍共享 user）\n")
        return True

    try:
        if stream_output:
            import json

            print("\nassistant> ", end="", flush=True)
            rag_note = ""
            result_text = ""
            async for payload in stream_chat_turn_events(
                session_handle,
                line,
                engine=engine,
                enable_rag=enable_rag,
                stream_mode="auto",
            ):
                event = json.loads(payload)
                if event.get("type") == "meta" and enable_rag:
                    rag_note = (
                        f"  (RAG: {event.get('evidence_count', 0)} 条证据"
                        f"{', 空' if event.get('rag_empty') else ''})"
                    )
                elif event.get("type") == "delta":
                    print(event.get("text", ""), end="", flush=True)
                elif event.get("type") == "done":
                    result_text = event.get("assistant_text") or ""
            print(f"{rag_note}\n")
            agent_debug(
                "TURN-OK",
                "chat_repl:_handle_line:stream",
                "流式对话完成",
                {"assistant_chars": len(result_text), "engine": engine},
            )
            if is_memory_runtime_debug() and memory_debug_console_enabled():
                st = await collect_memory_runtime_status(
                    run_ctx, event="turn_stream_done", verbose=True
                )
                print(
                    format_post_turn_debug(
                        st,
                        user_message=line,
                        evidence_count=0,
                        rag_empty=True,
                        history_turns=0,
                        assistant_chars=len(result_text),
                        layer_trigger_timeline=list(
                            (run_ctx.extra or {}).get("layer_triggers") or []
                        ),
                    )
                )
                print(format_layer_triggers(run_ctx))
            return True

        result = await execute_chat_turn(
            session_handle,
            line,
            engine=engine,
            enable_rag=enable_rag,
        )
    except Exception as exc:
        agent_debug(
            "ERR",
            "chat_repl:_handle_line",
            "对话轮次异常",
            {"error": str(exc), "engine": engine, "enable_rag": enable_rag},
        )
        print(f"\n[error] {exc}\n")
        return True

    agent_debug(
        "TURN-OK",
        "chat_repl:_handle_line",
        "对话轮次完成",
        {
            "engine": engine,
            "enable_rag": enable_rag,
            "assistant_chars": len(result.assistant_text or ""),
            "evidence_count": result.evidence_count,
            "rag_empty": result.rag_empty,
            "history_turns": result.history_turns,
        },
    )

    rag_note = ""
    if enable_rag:
        rag_note = (
            f"  (RAG: {result.evidence_count} 条证据"
            f"{', 空' if result.rag_empty else ''})"
        )
    print(f"\nassistant>{rag_note}\n{result.assistant_text}\n")
    if is_memory_runtime_debug() and memory_debug_console_enabled():
        st = await collect_memory_runtime_status(
            run_ctx, event="turn_console", verbose=True
        )
        print(
            format_post_turn_debug(
                st,
                user_message=line,
                evidence_count=result.evidence_count,
                rag_empty=result.rag_empty,
                history_turns=result.history_turns,
                assistant_chars=len(result.assistant_text or ""),
                layer_trigger_timeline=list(
                    (run_ctx.extra or {}).get("layer_triggers") or []
                ),
            )
        )
        print(format_layer_triggers(run_ctx))
    return True


async def chat_repl(
    tenant: str,
    user: str,
    session: str,
    *,
    data_dir: Path,
    config_dir: str,
    enable_rag: bool,
    engine: str,
    profile: str = "dev",
    debug: bool = False,
    debug_quiet: bool = False,
    no_debug: bool = False,
    enable_memory_tools: bool = True,
    stream_output: bool = False,
) -> int:
    trace_on, console_debug = resolve_memory_trace(
        profile=profile,
        debug=debug,
        debug_quiet=debug_quiet,
        no_debug=no_debug,
    )
    auto_trace = trace_on and not debug and not debug_quiet
    if debug_quiet or (trace_on and not console_debug):
        set_debug_quiet(trace_on)
    else:
        set_debug_console(console_debug)
    set_memory_runtime_debug(trace_on)
    set_memory_runtime_verbose(trace_on)
    chat_cfg = load_chat_config(config_dir)
    if not enable_memory_tools:
        chat_cfg = replace(chat_cfg, enable_memory_tools=False)
    request = RequestContext(
        tenant_id=tenant,
        user_id=user,
        session_id=session,
        trace_id="chat_repl",
        channel="cli",
    )
    run_ctx = build_chat_run_context(
        request,
        profile=profile,  # type: ignore[arg-type]
        config_dir=config_dir,
        data_dir=str(data_dir),
    )
    await run_ctx.require_memory().ensure_session(request)

    reg = run_ctx.models
    main_info = reg.get_model_info("main_llm") if reg else None
    main_role = run_ctx.models._roles.get("main_llm") if hasattr(run_ctx.models, "_roles") else None
    agent_debug(
        "STARTUP",
        "chat_repl:startup",
        "Agent 启动",
        {
            "tenant": tenant,
            "user": user,
            "session": session,
            "engine": engine,
            "profile": profile,
            "enable_rag": enable_rag,
            "debug": debug,
            "data_dir": str(data_dir),
            "rag_port": run_ctx.rag is not None,
            "memory_port": run_ctx.memory is not None,
            "rag_chroma_dir": (run_ctx.extra or {}).get("rag_chroma_dir"),
            "rag_tenant_id": (run_ctx.extra or {}).get("rag_tenant_id"),
            "main_llm_profile": main_info.profile if main_info else None,
            "main_llm_provider": main_info.provider if main_info else None,
            "main_llm_role_profile": (
                main_role.profile if main_role else None
            ),
            "main_llm_fallback": (
                list(main_role.fallback_chain or []) if main_role else []
            ),
        },
    )
    await log_memory_runtime_status(run_ctx, event="startup", console=debug)
    # #region agent log
    from app.agents.memory_runtime_debug import chat_config_verify_snapshot, trace_write

    trace_write(
        hypothesis_id="VERIFY-STARTUP",
        location="chat_repl:startup_config",
        message="round2 startup config snapshot",
        data={
            "chat_config": chat_config_verify_snapshot(chat_cfg),
            "enable_rag_cli": enable_rag,
            "engine": engine,
            "profile": profile,
        },
        run_id=session,
        force=True,
    )
    # #endregion

    session_handle = ChatSessionHandle(run_ctx=run_ctx, chat_cfg=chat_cfg)
    try:
        if engine == "langgraph":
            session_handle.lg_session = await create_chat_langgraph_session_async(
                run_ctx, chat_cfg=chat_cfg
            )

        print("对话 Agent（记忆 L1/L2 + RAG + 企业 Context）")
        print(f"  tenant={tenant}  user={user}  session={session}")
        print(
            f"  data_dir={data_dir}  rag={'on' if enable_rag else 'off'}"
            f"  engine={engine}  profile={profile}"
        )
        print(f"  memory_tools={'on' if chat_cfg.enable_memory_tools else 'off'}")
        mem_cfg = (run_ctx.extra or {}).get("memory_config_summary") or {}
        if mem_cfg:
            print(
                f"  memory: archive={mem_cfg.get('archive_backend')}  "
                f"cold={mem_cfg.get('enable_cold_archive')}"
            )
        if os.environ.get("MEMORY_CONFIG"):
            print(f"  MEMORY_CONFIG={os.environ.get('MEMORY_CONFIG')}")
        if stream_output:
            print("  stream=on")
        if trace_on:
            mode = "auto(dev)" if auto_trace else ("quiet" if debug_quiet else "verbose")
            console = "on" if memory_debug_console_enabled() else "off"
            cache_obj = (run_ctx.extra or {}).get("cache")
            cache_name = type(cache_obj).__name__ if cache_obj else "none"
            print(f"  memory_trace={mode}  console_detail={console}")
            print(f"  session_search_cache={cache_name}")
            print("  主日志: " + str(debug_log_path()))
            print("  细粒度: " + str(agent_debug_log_path()))
            print("  (关闭轮次控制台详情: MEMORY_DEBUG_CONSOLE=0)")
        print(HELP)

        while True:
            try:
                line = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                line = "/quit"
            if not line:
                continue
            cont = await _handle_line(
                line,
                run_ctx=session_handle.run_ctx,
                enable_rag=enable_rag,
                chat_cfg=chat_cfg,
                engine=engine,
                session_handle=session_handle,
                stream_output=stream_output,
                debug=debug,
            )
            if not cont:
                break

        await end_agent_session(session_handle.run_ctx, chat_cfg=chat_cfg)
        await log_memory_runtime_status(
            session_handle.run_ctx, event="session_end", console=debug
        )
        snap = session_handle.run_ctx.require_memory().compose_prompt_snapshot(
            session_handle.run_ctx.request
        )
        agent_debug(
            "FINALIZE",
            "chat_repl:end",
            "会话结束 L4→L1 finalize",
            {
                "hash": snap.hash,
                "memory_chars": len(snap.memory_text or ""),
                "memory_preview": (snap.memory_text or "")[:500],
            },
        )
        print("\n[done] session ended + L4 merged into L1")
        print(f"hash={snap.hash}\n")
        return 0
    finally:
        from app.runtime.adapters.langgraph.checkpointer import (
            teardown_postgres_checkpointer,
        )

        await teardown_postgres_checkpointer()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="记忆 + RAG 对话 REPL")
    parser.add_argument("--tenant", default="tenant1")
    parser.add_argument("--user", default="user1")
    parser.add_argument("--session", default="chat1")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="关闭 RAG 检索，仅使用记忆",
    )
    parser.add_argument(
        "--engine",
        choices=("langgraph", "direct"),
        default="langgraph",
        help="langgraph: prepare+react_agent+persist；direct: run_chat_turn",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="详细调试：L1-L4/RAG/Context 全链路 NDJSON（见 .cursor/debug-d0a1fc.log）",
    )
    parser.add_argument(
        "--debug-quiet",
        action="store_true",
        help="同 --debug，仅写日志文件，不打印控制台 debug 摘要",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="关闭 dev 默认自动 memory trace（仍可用 /debug-memory 手动快照）",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="关闭 Path B 记忆工具（session_search / remember_user_fact）",
    )
    parser.add_argument(
        "--profile",
        choices=("dev", "production"),
        default="dev",
        help="RunContext：dev 本地开发 | production 企业组件",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="流式打印 assistant 回复（LangGraph / direct token 流）",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(
        chat_repl(
            args.tenant,
            args.user,
            args.session,
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            enable_rag=not args.no_rag,
            engine=args.engine,
            profile=args.profile,
            debug=args.debug,
            debug_quiet=args.debug_quiet,
            no_debug=args.no_debug,
            enable_memory_tools=not args.no_tools,
            stream_output=args.stream,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
