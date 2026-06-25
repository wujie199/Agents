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
from app.agents.memory.memory_bootstrap import bootstrap_memory_runtime

from app.agents.orchestration.chat_config import load_chat_config
from app.agents.orchestration.chat_service import (
    ChatSessionHandle,
    execute_chat_turn,
    stream_chat_turn_events,
)
from app.agents.orchestration.chat_langgraph import create_chat_langgraph_session_async
from app.agents.memory.memory_metrics import get_memory_metric_stats, prometheus_text
from app.agents.memory.memory_views import list_pending_l1_deltas
from app.agents.memory.enterprise_memory import (
    confirm_pending_l1,
    format_finalize_summary,
    get_memory_status,
    list_user_sessions_enriched,
    refresh_l4_profile,
)
from app.agents.roles.react_loop import end_agent_session
from app.agents.debug.debug_trace import agent_debug, agent_debug_log_path, set_debug_console, set_debug_quiet

# DeepAgent 路由门控（可选依赖，enable=false 时不走）
from app.runtime.adapters.deepagents.config import load_deep_agent_config, DeepAgentConfig
from app.runtime.adapters.deepagents.routing_gate import should_use_deep_agent, should_use_deep_agent_async
from app.runtime.adapters.deepagents.adapter import DeepAgentAdapter
from app.runtime.adapters.deepagents import is_deep_agents_available
from app.agents.memory.memory_runtime_debug import (
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
  /refresh-profile          刷新 L4 外部画像缓存（不写入 L1）
  /cache-test [关键词]      连调两次 session_search 验证 L2 HIT（跨轮不清，/quit 才 CLEAR）
  /rag-cache-test [问题]    同一问题连调两次 RAG，验证 route_and_retrieve 缓存 HIT
  /redis-health             打印 Redis 缓存适配器 health + stats
  /metrics                  记忆子系统指标（ObservabilityPort）
  /todos                    查看 DeepAgent 规划任务列表（TodoList）
  /conflicts                查看当前 L1 冲突检测结果
  /decay-info               查看时间衰减配置和效果预览
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


def _print_pending_hint(run_ctx, pending_before: int) -> None:
    pending_after = list_pending_l1_deltas(run_ctx)
    if len(pending_after) > pending_before:
        rows = pending_after[-(len(pending_after) - pending_before) :]
        items = ", ".join(
            f"{r.get('key')}={r.get('value')}" for r in rows if r.get("key")
        )
        print(
            f"[HITL] 新增待确认记忆: {items or len(pending_after) - pending_before} 条。"
            "输入 /pending 查看，/confirm 确认写入 L1。\n"
        )


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
        pending = list_pending_l1_deltas(run_ctx)
        if not pending:
            print("\n（无 pending L1 记忆）\n")
            return True
        print("\n--- 即将确认 ---")
        for row in pending:
            print(f"  {row.get('key')} = {row.get('value')}  [{row.get('source')}]")
        n = await confirm_pending_l1(run_ctx)
        snap = memory.compose_prompt_snapshot(req)
        print(f"\n[ok] 已确认 {n} 条 → L1 hash={snap.hash}\n")
        return True

    if line == "/pending":
        pending = list_pending_l1_deltas(run_ctx)
        if not pending:
            print("\n（无 pending L1 记忆）\n")
            return True
        print("\n--- pending L1（输入 /confirm 写入）---")
        for row in pending:
            print(f"  {row.get('key')} = {row.get('value')}  [{row.get('source')}]")
        print()
        return True

    if line == "/sessions":
        rows = await list_user_sessions_enriched(run_ctx, limit=20)
        if not rows:
            print("\n（无历史 session）\n")
            return True
        print("\n--- sessions (online + cold) ---")
        for row in rows:
            storage = row.get("storage") or "online"
            extra = ""
            if row.get("message_count") is not None:
                extra = f"  msgs={row.get('message_count')}"
            print(
                f"  {row.get('session_id')}  storage={storage}  "
                f"status={row.get('status')}  "
                f"started={row.get('started_at', '')}{extra}"
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

    if line == "/refresh-profile":
        try:
            result = await refresh_l4_profile(run_ctx)
        except Exception as exc:
            print(f"\n[error] L4 refresh failed: {exc}\n")
            return True
        facts = result.get("facts") or []
        print(f"\n--- L4 refreshed: {result.get('fact_count', 0)} facts ---")
        for row in facts[:8]:
            if isinstance(row, dict):
                print(f"  {row.get('key')} = {row.get('value')}  [{row.get('source')}]")
        print()
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
        print(
            "  (连续两次 session_search → HIT；persist_turn 不清缓存，/quit finalize 才 CLEAR)\n"
        )
        r1 = await memory.session_search(
            probe_query, req, limit=5, scope="session"
        )
        r2 = await memory.session_search(
            probe_query, req, limit=5, scope="session"
        )
        print(f"  1st: len={len(r1)}  preview={r1[:120]!r}")
        print(f"  2nd: len={len(r2)}  preview={r2[:120]!r}")
        print(f"  identical={r1 == r2}")
        print(f"  → 日志: {debug_log_path()} (MEMORY_RUNTIME_DEBUG=1 时写入)\n")
        return True

    if line == "/redis-health":
        cache = (run_ctx.extra or {}).get("cache")
        adapter = type(cache).__name__ if cache else "none"
        print("\n--- Redis cache health ---")
        print(f"  cache_adapter={adapter}")
        if cache is None:
            print("  (无 cache port)\n")
            return True
        if hasattr(cache, "health"):
            try:
                health = await cache.health()
                for key, val in health.items():
                    print(f"  {key}={val}")
            except Exception as exc:
                print(f"  health() failed: {exc}")
        elif hasattr(cache, "get_stats"):
            print(f"  stats={cache.get_stats()}")
        else:
            print("  (adapter 无 health/get_stats)")
        print(f"  → NDJSON: {debug_log_path()}\n")
        return True

    if line.startswith("/rag-cache-test"):
        if not enable_rag:
            print("\n[RAG disabled] 请去掉 --no-rag 启动\n")
            return True
        import time

        from app.agents.orchestration.chat_nodes import retrieve_rag_bundle

        parts = line.split(maxsplit=1)
        probe_query = (
            parts[1].strip() if len(parts) > 1 else "如何选择扫地机器人"
        )
        print("\n--- RAG cache probe ---")
        print(f"  query={probe_query!r}")
        print("  (连续两次 route_and_retrieve，第 2 次应 REDIS HIT)\n")
        t0 = time.perf_counter()
        bundle1 = await retrieve_rag_bundle(run_ctx, probe_query)
        t1 = time.perf_counter()
        bundle2 = await retrieve_rag_bundle(run_ctx, probe_query)
        t2 = time.perf_counter()
        ms1 = round((t1 - t0) * 1000)
        ms2 = round((t2 - t1) * 1000)
        n1 = len(bundle1.evidences or [])
        n2 = len(bundle2.evidences or [])
        print(
            f"  1st: {n1} evidences  {ms1}ms  empty={bundle1.empty}"
        )
        print(
            f"  2nd: {n2} evidences  {ms2}ms  empty={bundle2.empty}"
        )
        print(
            f"  → 日志: {debug_log_path()} "
            f"(MEMORY_RUNTIME_DEBUG=1 时写入)\n"
        )
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

    if line == "/todos":
        da_config = load_deep_agent_config()
        if not da_config.enable_deep_agent:
            print("\n（DeepAgent 未启用，chat.yml 中 deep_agent.enable=true 开启）\n")
            return True
        if not is_deep_agents_available():
            print("\n（deepagents 未安装，pip install 'agents[planning]'）\n")
            return True
        try:
            da_adapter = DeepAgentAdapter(run_ctx, da_config)
            todos = await da_adapter.get_todos()
            if not todos:
                print("\n（无规划任务）\n")
                return True
            print("\n--- DeepAgent TodoList ---")
            for i, t in enumerate(todos, 1):
                status = t.get("status", "?")
                desc = t.get("description", "")
                deps = t.get("depends_on") or []
                dep_str = f"  ← {','.join(str(d) for d in deps)}" if deps else ""
                print(f"  {i}. [{status}] {desc}{dep_str}")
            print()
        except Exception as exc:
            print(f"\n[error] 获取 TodoList 失败: {exc}\n")
        return True

    if line == "/conflicts":
        from app.agents.memory.conflict_detector import (
            check_l1_write_conflicts,
            is_values_conflicting,
            ConflictStrategy,
        )
        memory = run_ctx.require_memory()
        pending = list_pending_l1_deltas(run_ctx)
        if not pending:
            print("\n（无 pending L1，无冲突可检测）\n")
            return True
        try:
            snap = memory.compose_prompt_snapshot(req)
            existing_facts: dict[str, str] = {}
            for ln in (snap.memory_text or "").split("\n"):
                ln = ln.strip()
                if ": " in ln:
                    k, v = ln.split(": ", 1)
                    existing_facts[k.strip()] = v.strip()
        except Exception:
            existing_facts = {}
        records = check_l1_write_conflicts(
            existing_facts,
            [{"key": d.get("key"), "value": d.get("value")} for d in pending],
            strategy=ConflictStrategy(chat_cfg.l1_conflict_strategy),
            l1_auto_write_confidence_min=chat_cfg.l1_auto_write_confidence_min,
        )
        conflicts = [r for r in records if is_values_conflicting(r.old_value, r.new_value)]
        if not conflicts:
            print(f"\n--- L1 冲突检测：{len(records)} 条 pending，无冲突 ---\n")
            return True
        print(f"\n--- L1 冲突检测：{len(conflicts)}/{len(records)} 条冲突 ---")
        for r in conflicts:
            print(f"  [{r.strategy}] {r.key}: '{r.old_value}' → '{r.new_value}' → resolved='{r.resolved_value}' (HITL={r.needs_hitl})")
        print()
        return True

    if line == "/decay-info":
        from agent_platform.memory.adapters.time_decay import time_decay_factor
        from datetime import datetime, timezone, timedelta
        print(f"\n--- 时间衰减配置 ---")
        print(f"  enabled: {chat_cfg.time_decay}")
        print(f"  half_life_days: {chat_cfg.time_decay_half_life_days}")
        if chat_cfg.time_decay:
            now = datetime.now(timezone.utc)
            print(f"\n  衰减预览（half_life={chat_cfg.time_decay_half_life_days}天）:")
            for days in (1, 7, 30, 90, 180, 365):
                ts = (now - timedelta(days=days)).isoformat()
                factor = time_decay_factor(ts, now=now, half_life_days=chat_cfg.time_decay_half_life_days)
                print(f"    {days:>3}天前 → 衰减因子 {factor:.4f}")
        print()
        return True

    if line == "/new" or line.startswith("/new "):
        import uuid
        from dataclasses import replace

        from app.agents.orchestration.chat_langgraph import create_chat_langgraph_session_async

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
            from app.agents.orchestration.chat_langgraph import create_chat_langgraph_session_async

            session_handle.lg_session = await create_chat_langgraph_session_async(
                session_handle.run_ctx, chat_cfg=chat_cfg
            )
        await memory.ensure_session(new_req)
        print(f"\n→ 新会话 session_id={new_sid}（L2 历史独立，L1 仍共享 user）\n")
        return True

    pending_before = len(list_pending_l1_deltas(run_ctx))

    # ── DeepAgent 路由门控 ──
    da_config = load_deep_agent_config()
    if should_use_deep_agent(line, chat_cfg, da_config):
        try:
            from app.runtime.adapters.deepagents.subagent_bridge import InnerSubAgentBridge
            compiled = getattr(session_handle, "lg_session", None)
            if compiled is not None and is_deep_agents_available():
                bridge = InnerSubAgentBridge(
                    compiled_graph=compiled,
                    runtime=None,
                    name="chat-worker",
                    description="执行单轮对话",
                )
                da_adapter = DeepAgentAdapter(run_ctx, da_config, inner_bridge=bridge)
                da_result = await da_adapter.invoke(line)
                da_text = da_result.get("assistant_text") or ""
                da_todos = da_result.get("todos") or []
                if da_text:
                    print(f"\nassistant>  (DeepAgent 规划层)\n{da_text}\n")
                if da_todos:
                    print(f"  [todos] {len(da_todos)} 个任务:")
                    for t in da_todos:
                        status = t.get("status", "?")
                        desc = t.get("description", "")
                        print(f"    [{status}] {desc}")
                    print()
                if not da_text and not da_todos:
                    print("\n[DeepAgent] 规划完成但无输出\n")
                _print_pending_hint(run_ctx, pending_before)
                return True
            else:
                print(f"\n[DeepAgent] 已触发但 deepagents 未安装或会话未初始化，回退内层图\n")
        except Exception as exc:
            print(f"\n[DeepAgent error] {exc}，回退内层图\n")

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
            _print_pending_hint(run_ctx, pending_before)
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
    _print_pending_hint(run_ctx, pending_before)
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
    chat_cfg = load_chat_config(config_dir, profile=profile)
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
    boot = await bootstrap_memory_runtime(
        run_ctx,
        data_dir=str(data_dir),
        config_dir=config_dir,
        profile=profile,
    )
    agent_debug(
        "STARTUP",
        "chat_repl:bootstrap",
        "记忆 bootstrap",
        boot,
    )

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
    if trace_on:
        from app.agents.memory.memory_runtime_debug import (
            chat_config_verify_snapshot,
            trace_write,
        )

        trace_write(
            hypothesis_id="VERIFY-STARTUP",
            location="chat_repl:startup_config",
            message="startup config snapshot",
            data={
                "chat_config": chat_config_verify_snapshot(chat_cfg),
                "enable_rag_cli": enable_rag,
                "engine": engine,
                "profile": profile,
            },
            run_id=session,
        )

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

        if chat_cfg.auto_confirm_pending_on_exit:
            pending = list_pending_l1_deltas(session_handle.run_ctx)
            if pending:
                n = await confirm_pending_l1(session_handle.run_ctx)
                print(f"\n[auto-confirm] 已确认 {n} 条 pending L1\n")

        fin_summary = await end_agent_session(
            session_handle.run_ctx, chat_cfg=chat_cfg
        )
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
                "finalize_summary": fin_summary,
            },
        )
        print("\n[done] session ended")
        print(f"  {format_finalize_summary(fin_summary)}")
        print(f"  L1 hash={snap.hash}\n")
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
        help="详细调试：L1-L4/RAG/Context 全链路（MEMORY_RUNTIME_DEBUG + AGENT_PIPELINE_DEBUG）",
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
