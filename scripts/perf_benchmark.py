#!/usr/bin/env python3
"""性能基准测试：启动 Agent、发送消息、采集 NDJSON 日志、解析精确耗时。

用法:
  python scripts/perf_benchmark.py                     # 单轮默认消息
  python scripts/perf_benchmark.py -m "你好"            # 自定义消息
  python scripts/perf_benchmark.py -m "你好" "介绍一下自己"  # 多轮
  python scripts/perf_benchmark.py --no-rag             # 关闭 RAG
  python scripts/perf_benchmark.py --rounds 3           # 重复 3 轮取平均
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import replace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.domain.context import RequestContext
from app.agents.context_factory import build_chat_run_context
from app.agents.memory.memory_bootstrap import bootstrap_memory_runtime
from app.agents.orchestration.chat_config import load_chat_config
from app.agents.orchestration.chat_service import (
    ChatSessionHandle,
    stream_chat_turn_events,
)
from app.agents.orchestration.chat_langgraph import create_chat_langgraph_session_async
from app.agents.memory.memory_runtime_debug import (
    debug_log_path,
    set_memory_runtime_debug,
    set_memory_runtime_verbose,
    resolve_memory_trace,
)
from app.agents.debug.debug_trace import (
    agent_debug,
    agent_debug_log_path,
    set_debug_console,
    set_debug_quiet,
)
from app.agents.roles.react_loop import end_agent_session
from app.agents.memory.memory_views import list_pending_l1_deltas
from app.agents.memory.enterprise_memory import confirm_pending_l1


def _clear_ndjson_logs():
    """清空已有 NDJSON 日志文件，确保本次采集干净。"""
    for p in (debug_log_path(), agent_debug_log_path()):
        if p.exists():
            p.write_text("", encoding="utf-8")


def _parse_ndjson(path: Path) -> list[dict]:
    """解析 NDJSON 文件为 dict 列表。"""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _extract_perf_stages(records: list[dict]) -> list[dict]:
    """从 NDJSON 记录中提取 PERF-LATENCY 条目。"""
    stages = []
    for r in records:
        if r.get("hypothesisId") == "PERF-LATENCY":
            data = r.get("data") or {}
            stages.append(data)
    return stages


def _extract_perf_summary(records: list[dict]) -> list[dict]:
    """提取 PERF-SUMMARY 条目。"""
    summaries = []
    for r in records:
        if r.get("hypothesisId") == "PERF-SUMMARY":
            summaries.append(r.get("data") or {})
    return summaries


def _extract_node_timing(records: list[dict]) -> list[dict]:
    """提取 NODE-TIMING 条目。"""
    timings = []
    for r in records:
        if r.get("hypothesisId") == "NODE-TIMING":
            timings.append(r.get("data") or {})
    return timings


def _extract_node_timing_summary(records: list[dict]) -> list[dict]:
    """提取 NODE-TIMING-SUMMARY 条目。"""
    summaries = []
    for r in records:
        if r.get("hypothesisId") == "NODE-TIMING-SUMMARY":
            summaries.append(r.get("data") or {})
    return summaries


def _print_report(
    perf_stages: list[dict],
    perf_summaries: list[dict],
    node_timings: list[dict],
    node_timing_summaries: list[dict],
    turn_wall_ms: list[float],
):
    """打印可读的性能报告。"""
    print("\n" + "=" * 72)
    print("  性能日志精确分析报告")
    print("=" * 72)

    # ── 1. 轮次墙钟总耗时 ──
    if turn_wall_ms:
        print(f"\n── 轮次墙钟耗时 ──")
        for i, ms in enumerate(turn_wall_ms, 1):
            print(f"  Round {i}: {ms:.0f} ms  ({ms/1000:.2f} s)")
        if len(turn_wall_ms) > 1:
            avg = sum(turn_wall_ms) / len(turn_wall_ms)
            print(f"  平均: {avg:.0f} ms  ({avg/1000:.2f} s)")

    # ── 2. 节点级 TimingMiddleware 瀑布图 ──
    if node_timing_summaries:
        print(f"\n── TimingMiddleware 瀑布图汇总 ──")
        for i, s in enumerate(node_timing_summaries, 1):
            print(f"\n  Round {i}:  total={s.get('total_ms', 0):.1f} ms")
            nodes = s.get("nodes") or []
            for n in nodes:
                bar_len = int(n.get("pct", 0) / 2)
                bar = "█" * bar_len + "░" * max(0, 50 - bar_len)
                print(
                    f"    {n.get('node', '?'):>10}  {n.get('duration_ms', 0):>8.1f} ms  "
                    f"({n.get('pct', 0):>5.1f}%)  {bar}"
                )
            slow = s.get("slow_nodes") or []
            if slow:
                print(f"    ⚠ 慢节点: {', '.join(slow)}")

    # ── 3. 细粒度 PERF-LATENCY 阶段耗时 ──
    if perf_stages:
        print(f"\n── 细粒度阶段耗时（perf_mark）──")
        # 按耗时降序
        sorted_stages = sorted(perf_stages, key=lambda s: s.get("duration_ms", 0), reverse=True)
        total_accounted = sum(s.get("duration_ms", 0) for s in sorted_stages)
        print(f"  已统计总耗时: {total_accounted:.1f} ms")
        print()
        for s in sorted_stages:
            stage = s.get("stage", "?")
            ms = s.get("duration_ms", 0)
            pct = (ms / total_accounted * 100) if total_accounted > 0 else 0
            extra_parts = []
            for k, v in s.items():
                if k not in ("stage", "duration_ms"):
                    extra_parts.append(f"{k}={v}")
            extra_str = f"  [{', '.join(extra_parts)}]" if extra_parts else ""
            print(f"  {stage:>40}  {ms:>8.1f} ms  ({pct:>5.1f}%){extra_str}")

    # ── 4. PERF-SUMMARY 汇总 ──
    if perf_summaries:
        print(f"\n── PERF-SUMMARY 汇总 ──")
        for i, s in enumerate(perf_summaries, 1):
            print(f"\n  Round {i}:  phase={s.get('phase')}  label={s.get('label')}")
            print(f"    total_ms       = {s.get('total_ms', 0):.1f}")
            print(f"    accounted_ms   = {s.get('accounted_ms', 0):.1f}")
            print(f"    unaccounted_ms = {s.get('unaccounted_ms', 0):.1f}")
            slowest = s.get("slowest") or []
            if slowest:
                print(f"    耗时 Top8:")
                for row in slowest:
                    print(
                        f"      · {row.get('stage', '?'):>40}  "
                        f"{row.get('duration_ms', 0):>8.1f} ms"
                    )

    # ── 5. LLM 关键指标提取 ──
    llm_stages = [s for s in perf_stages if s.get("stage", "").startswith("LLM.")]
    if llm_stages:
        print(f"\n── LLM 关键指标 ──")
        for s in llm_stages:
            stage = s.get("stage")
            ms = s.get("duration_ms", 0)
            extra = {k: v for k, v in s.items() if k not in ("stage", "duration_ms")}
            print(f"  {stage}: {ms:.1f} ms  {extra}")

    # ── 6. 节点级 TimingMiddleware 明细 ──
    if node_timings:
        print(f"\n── 节点级 TimingMiddleware 明细 ──")
        for t in node_timings:
            node = t.get("node", "?")
            ms = t.get("duration_ms", 0)
            slow = "⚠ SLOW" if t.get("slow") else ""
            err = t.get("error") or ""
            print(f"  {node:>10}  {ms:>8.1f} ms  {slow} {err}")

    print("\n" + "=" * 72)


async def _run_single_turn(
    session_handle: ChatSessionHandle,
    user_message: str,
    *,
    enable_rag: bool,
    engine: str,
    chat_cfg,
) -> float:
    """执行单轮对话，返回墙钟耗时(ms)。"""
    t0 = time.perf_counter()
    parts: list[str] = []
    async for payload in stream_chat_turn_events(
        session_handle,
        user_message,
        engine=engine,
        enable_rag=enable_rag,
        stream_mode="auto",
    ):
        event = json.loads(payload)
        if event.get("type") == "delta":
            parts.append(event.get("text", ""))
        elif event.get("type") == "done":
            pass
    wall_ms = (time.perf_counter() - t0) * 1000
    return wall_ms


async def perf_benchmark(
    messages: list[str],
    *,
    tenant: str,
    user: str,
    session: str,
    data_dir: Path,
    config_dir: str,
    enable_rag: bool,
    engine: str,
    profile: str,
    rounds: int,
) -> None:
    # ── 开启全量 debug 日志 ──
    trace_on, console_debug = resolve_memory_trace(
        profile=profile, debug=True, debug_quiet=False, no_debug=False,
    )
    set_debug_quiet(True)   # 日志写文件，不打印控制台
    set_memory_runtime_debug(True)
    set_memory_runtime_verbose(True)

    request = RequestContext(
        tenant_id=tenant,
        user_id=user,
        session_id=session,
        trace_id="perf_benchmark",
        channel="cli",
    )
    run_ctx = build_chat_run_context(
        request,
        profile=profile,
        config_dir=config_dir,
        data_dir=str(data_dir),
    )
    boot = await bootstrap_memory_runtime(
        run_ctx,
        data_dir=str(data_dir),
        config_dir=config_dir,
        profile=profile,
    )
    chat_cfg = load_chat_config(config_dir, profile=profile)

    session_handle = ChatSessionHandle(run_ctx=run_ctx, chat_cfg=chat_cfg)
    if engine == "langgraph":
        session_handle.lg_session = await create_chat_langgraph_session_async(
            run_ctx, chat_cfg=chat_cfg
        )

    print(f"性能基准测试")
    print(f"  tenant={tenant}  user={user}  session={session}")
    print(f"  rag={'on' if enable_rag else 'off'}  engine={engine}  profile={profile}")
    print(f"  messages={messages}  rounds={rounds}")
    print(f"  NDJSON: {debug_log_path()}")
    print(f"  trace:  {agent_debug_log_path()}")

    # ── 执行多轮 ──
    all_wall_ms: list[float] = []

    for round_idx in range(rounds):
        for msg_idx, msg in enumerate(messages):
            # 每轮前清空日志，确保数据干净
            _clear_ndjson_logs()

            round_label = f"R{round_idx+1}-M{msg_idx+1}"
            print(f"\n── {round_label}: \"{msg}\" ──")
            wall_ms = await _run_single_turn(
                session_handle,
                msg,
                enable_rag=enable_rag,
                engine=engine,
                chat_cfg=chat_cfg,
            )
            all_wall_ms.append(wall_ms)
            print(f"  墙钟耗时: {wall_ms:.0f} ms ({wall_ms/1000:.2f} s)")

            # ── 解析本轮日志 ──
            ndjson_records = _parse_ndjson(debug_log_path())
            trace_records = _parse_ndjson(agent_debug_log_path())
            # agent_debug 镜像到 NDJSON，所以 ndjson_records 包含全部
            all_records = ndjson_records

            perf_stages = _extract_perf_stages(all_records)
            perf_summaries = _extract_perf_summary(all_records)
            node_timings = _extract_node_timing(all_records)
            node_timing_summaries = _extract_node_timing_summary(all_records)

            _print_report(
                perf_stages,
                perf_summaries,
                node_timings,
                node_timing_summaries,
                [wall_ms],
            )

    # ── 如果多轮多消息，打印汇总 ──
    if len(all_wall_ms) > 1:
        print(f"\n{'='*72}")
        print(f"  汇总（{len(all_wall_ms)} 次对话）")
        print(f"{'='*72}")
        for i, ms in enumerate(all_wall_ms):
            print(f"  #{i+1}: {ms:.0f} ms ({ms/1000:.2f} s)")
        avg = sum(all_wall_ms) / len(all_wall_ms)
        mn = min(all_wall_ms)
        mx = max(all_wall_ms)
        print(f"  平均: {avg:.0f} ms  最小: {mn:.0f} ms  最大: {mx:.0f} ms")

    # ── 结束会话 ──
    if chat_cfg.auto_confirm_pending_on_exit:
        pending = list_pending_l1_deltas(session_handle.run_ctx)
        if pending:
            await confirm_pending_l1(session_handle.run_ctx)
    await end_agent_session(session_handle.run_ctx, chat_cfg=chat_cfg)

    from app.runtime.adapters.langgraph.checkpointer import teardown_postgres_checkpointer
    await teardown_postgres_checkpointer()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="性能基准测试")
    parser.add_argument("-m", "--messages", nargs="+", default=["你好，请介绍一下你自己"])
    parser.add_argument("--tenant", default="tenant1")
    parser.add_argument("--user", default="user1")
    parser.add_argument("--session", default="perf-bench")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--engine", choices=("langgraph", "direct"), default="langgraph")
    parser.add_argument("--profile", choices=("dev", "production"), default="dev")
    parser.add_argument("--rounds", type=int, default=1, help="重复轮次（每轮清空日志重新采集）")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(
        perf_benchmark(
            args.messages,
            tenant=args.tenant,
            user=args.user,
            session=args.session,
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            enable_rag=not args.no_rag,
            engine=args.engine,
            profile=args.profile,
            rounds=args.rounds,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
