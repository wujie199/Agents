# -*- coding: utf-8 -*-
"""TimingMiddleware：节点级性能监控，记录每个图节点的耗时、时间线与慢节点告警。

输出通道：
  1. logging（结构化 JSON） — 生产环境 ELK 采集
  2. trace_write（NDJSON）  — 开发调试 .cursor/memory_runtime.ndjson
  3. run_ctx.extra["node_timing"] — 下游 SSE/meta 事件可携带

最后一个节点退出时，自动输出瀑布图汇总（node_timing_summary）。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from core.composition.run_context import RunContext

_logger = logging.getLogger("app.agents.middleware.timing")

# ── 瀑布图汇总触发节点（图末端） ──
_SUMMARY_NODES = frozenset({"persist"})


class TimingMiddleware:
    """节点级性能监控 middleware：记录每个图节点的耗时、时间线、告警。"""

    def __init__(
        self,
        *,
        slow_threshold_ms: float = 3000,
        node_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """Args:
            slow_threshold_ms: 默认慢节点阈值（毫秒）。
            node_thresholds: 按节点名自定义阈值，如 {"agent": 5000, "persist": 1000}。
        """
        self._slow_threshold_ms = slow_threshold_ms
        self._node_thresholds = node_thresholds or {}

    @property
    def name(self) -> str:
        return "timing"

    # ── on_enter ──

    async def on_enter(
        self,
        node_name: str,
        state: Any,
        config: Any,
    ) -> Dict[str, Any]:
        return {
            "_timing_start": time.perf_counter(),
            "_timing_wall_start": time.time(),
        }

    # ── on_exit ──

    async def on_exit(
        self,
        node_name: str,
        state: Any,
        config: Any,
        result: Any,
        *,
        error: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = extra or {}
        t0 = ctx.get("_timing_start")
        wall_t0 = ctx.get("_timing_wall_start")

        if t0 is None:
            return

        # ── 1. 计算耗时 ──
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        wall_start_ms = round(wall_t0 * 1000) if wall_t0 else None
        threshold = self._node_thresholds.get(node_name, self._slow_threshold_ms)
        is_slow = duration_ms >= threshold

        # ── 2. 从 config 提取 run_ctx ──
        configurable = (config or {}).get("configurable") or {}
        run_ctx: RunContext | None = configurable.get("run_ctx")
        trace_id = ctx.get("trace_id", "")
        span_id = ctx.get("span_id", "")

        # ── 3. 构建 timing entry ──
        entry: Dict[str, Any] = {
            "event": "node_timing",
            "node": node_name,
            "duration_ms": duration_ms,
            "wall_start_ms": wall_start_ms,
            "trace_id": trace_id,
            "span_id": span_id,
            "error": str(error) if error else None,
            "slow": is_slow,
        }

        # ── 4. 写入结构化日志 ──
        if is_slow:
            _logger.warning(
                "node_timing node=%s duration_ms=%.2f slow=True threshold=%.0f trace_id=%s",
                node_name, duration_ms, threshold, trace_id,
            )
        else:
            _logger.debug(
                "node_timing node=%s duration_ms=%.2f trace_id=%s",
                node_name, duration_ms, trace_id,
            )

        # ── 5. 写入 NDJSON ──
        self._write_ndjson(entry, run_ctx)

        # ── 6. 累积到 run_ctx.extra["node_timing"] ──
        node_timing_list = self._accumulate(run_ctx, entry)

        # ── 7. 慢节点告警 ──
        if is_slow:
            _logger.warning(
                "SLOW_NODE node=%s duration_ms=%.2f threshold=%.0f",
                node_name, duration_ms, threshold,
            )

        # ── 8. 瀑布图汇总（末端节点） ──
        if node_name in _SUMMARY_NODES and node_timing_list:
            self._emit_summary(node_timing_list, trace_id, run_ctx)

    # ── 内部方法 ──

    @staticmethod
    def _accumulate(
        run_ctx: RunContext | None,
        entry: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        """累积 timing entry 到 run_ctx.extra["node_timing"]。"""
        if run_ctx is None or not isinstance(getattr(run_ctx, "extra", None), dict):
            return []
        timing_list: list = run_ctx.extra.setdefault("node_timing", [])
        timing_list.append(entry)
        return timing_list

    @staticmethod
    def _write_ndjson(entry: Dict[str, Any], run_ctx: RunContext | None) -> None:
        """写入 NDJSON 调试日志。"""
        try:
            from app.agents.memory.memory_runtime_debug import trace_write

            run_id = "default"
            if run_ctx is not None and getattr(run_ctx, "request", None) is not None:
                run_id = getattr(run_ctx.request, "session_id", None) or "default"

            trace_write(
                hypothesis_id="NODE-TIMING",
                location=f"timing.{entry.get('node', 'unknown')}",
                message="node timing",
                data=entry,
                run_id=run_id,
            )
        except Exception:
            pass

    def _emit_summary(
        self,
        timing_list: list[Dict[str, Any]],
        trace_id: str,
        run_ctx: RunContext | None,
    ) -> None:
        """输出瀑布图汇总。"""
        nodes = []
        total_ms = 0.0
        slow_nodes: list[str] = []

        for t in timing_list:
            dur = t.get("duration_ms", 0.0)
            total_ms += dur
            nodes.append({
                "node": t.get("node"),
                "duration_ms": dur,
            })
            if t.get("slow"):
                slow_nodes.append(t.get("node", ""))

        # 计算占比
        for n in nodes:
            n["pct"] = round(n["duration_ms"] / total_ms * 100, 1) if total_ms > 0 else 0.0

        summary: Dict[str, Any] = {
            "event": "node_timing_summary",
            "total_ms": round(total_ms, 2),
            "nodes": nodes,
            "slow_nodes": slow_nodes,
            "trace_id": trace_id,
        }

        _logger.debug(
            "node_timing_summary total_ms=%.2f nodes=%s slow=%s",
            total_ms,
            [n["node"] for n in nodes],
            slow_nodes,
        )

        # 写入 NDJSON
        try:
            from app.agents.memory.memory_runtime_debug import trace_write

            run_id = "default"
            if run_ctx is not None and getattr(run_ctx, "request", None) is not None:
                run_id = getattr(run_ctx.request, "session_id", None) or "default"

            trace_write(
                hypothesis_id="NODE-TIMING-SUMMARY",
                location="timing.summary",
                message="node timing summary (waterfall)",
                data=summary,
                run_id=run_id,
            )
        except Exception:
            pass

        # 存入 run_ctx.extra 供下游使用
        if run_ctx is not None and isinstance(getattr(run_ctx, "extra", None), dict):
            run_ctx.extra["node_timing_summary"] = summary
