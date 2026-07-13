# -*- coding: utf-8
"""Memory evaluation pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.composition.run_context import RunContext
from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta, TurnRecord

from app.agents.context_builder import extract_l1_facts_from_session
from app.agents.orchestration.chat_config import ChatAgentConfig, load_chat_config
from document.memory.evaluation.dataset import MemoryEvalSample, load_memory_eval_dataset
from document.memory.evaluation.metrics import (
    keyword_recall,
    kv_match_score,
    user_file_kv,
    user_snapshot_kv,
)
from document.memory.evaluation.report import write_memory_eval_reports


@dataclass
class MemoryEvalRow:
    id: str
    kind: str
    passed: bool = False
    score: float = 0.0
    detail: str = ""
    latency_ms: float = 0.0
    tags: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "passed": self.passed,
            "score": round(self.score, 4),
            "detail": self.detail[:500],
            "latency_ms": round(self.latency_ms, 2),
            "tags": self.tags,
            "error": self.error,
        }


async def _seed_session(memory: Any, sample: MemoryEvalSample) -> None:
    req = RequestContext(
        tenant_id=sample.tenant_id,
        user_id=sample.user_id,
        session_id=sample.session_id,
        trace_id=f"eval-{sample.id}",
        channel="memory_eval",
    )
    await memory.ensure_session(req)
    for turn in sample.seed_turns:
        role = turn.get("role") or "user"
        content = turn.get("content") or ""
        if content:
            await memory.persist_turn(
                req,
                TurnRecord(role=role, content=content, trace_id=req.trace_id),
            )


async def eval_session_search_row(
    memory: Any,
    sample: MemoryEvalSample,
    *,
    min_score: float = 0.5,
) -> MemoryEvalRow:
    t0 = time.perf_counter()
    row = MemoryEvalRow(id=sample.id, kind=sample.kind, tags=sample.tags)
    try:
        await _seed_session(memory, sample)
        req = RequestContext(
            tenant_id=sample.tenant_id,
            user_id=sample.user_id,
            session_id=sample.session_id,
            trace_id=f"eval-{sample.id}",
            channel="memory_eval",
        )
        summary = await memory.session_search(
            sample.query,
            req,
            limit=5,
            scope="session",
            mode="discovery",
        )
        score = keyword_recall(summary, sample.expected_keywords)
        row.score = score
        row.passed = score >= min_score
        row.detail = (summary or "")[:400]
    except Exception as exc:
        row.error = str(exc)
        row.passed = False
        row.score = 0.0
    row.latency_ms = (time.perf_counter() - t0) * 1000
    return row


async def eval_l1_extract_row(
    ctx: RunContext,
    sample: MemoryEvalSample,
    chat_cfg: ChatAgentConfig,
    *,
    min_score: float = 1.0,
) -> MemoryEvalRow:
    t0 = time.perf_counter()
    row = MemoryEvalRow(id=sample.id, kind=sample.kind, tags=sample.tags)
    try:
        turns = []
        if sample.transcript:
            for line in sample.transcript.splitlines():
                line = line.strip()
                if line.lower().startswith("user:"):
                    turns.append({"role": "user", "content": line[5:].strip()})
                elif line.lower().startswith("assistant:"):
                    turns.append({"role": "assistant", "content": line[10:].strip()})
        elif sample.seed_turns:
            turns = sample.seed_turns
        facts = await extract_l1_facts_from_session(ctx, turns, chat_cfg)
        extracted = {f["key"]: f["value"] for f in facts}
        score = kv_match_score(extracted, sample.expected_kv)
        row.score = score
        row.passed = score >= min_score
        row.detail = json.dumps(extracted, ensure_ascii=False)
    except Exception as exc:
        row.error = str(exc)
        row.passed = False
    row.latency_ms = (time.perf_counter() - t0) * 1000
    return row


async def eval_hitl_finalize_row(
    memory: Any,
    sample: MemoryEvalSample,
    *,
    min_score: float = 1.0,
) -> MemoryEvalRow:
    t0 = time.perf_counter()
    row = MemoryEvalRow(id=sample.id, kind=sample.kind, tags=sample.tags)
    try:
        req = RequestContext(
            tenant_id=sample.tenant_id,
            user_id=sample.user_id,
            session_id=sample.session_id,
            trace_id=f"eval-{sample.id}",
            channel="memory_eval",
        )
        for item in sample.pending_deltas:
            await memory.update_prompt_memory(
                req,
                MemoryDelta(
                    key=str(item["key"]),
                    value=str(item["value"]),
                    source=str(item.get("source") or "user"),
                ),
                require_hitl=True,
            )
        if sample.confirm_before_finalize:
            confirm = getattr(memory, "confirm_pending_deltas", None)
            if callable(confirm):
                await confirm(req)
        summary = await memory.finalize_session(req)
        snap = memory.compose_prompt_snapshot(req)
        kv = user_snapshot_kv(snap.memory_text)
        hot = getattr(memory, "_hot", None)
        if hot is not None:
            kv.update(user_file_kv(hot.get_raw_user(req.tenant_id, req.user_id)))
        score = kv_match_score(kv, sample.expected_kv)
        row.score = score
        row.passed = score >= min_score
        row.detail = json.dumps(
            {"kv": kv, "finalize": summary}, ensure_ascii=False
        )
    except Exception as exc:
        row.error = str(exc)
        row.passed = False
    row.latency_ms = (time.perf_counter() - t0) * 1000
    return row


async def eval_l4_merge_row(
    memory: Any,
    sample: MemoryEvalSample,
    *,
    min_score: float = 1.0,
) -> MemoryEvalRow:
    t0 = time.perf_counter()
    row = MemoryEvalRow(id=sample.id, kind=sample.kind, tags=sample.tags)
    try:
        req = RequestContext(
            tenant_id=sample.tenant_id,
            user_id=sample.user_id,
            session_id=sample.session_id,
            trace_id=f"eval-{sample.id}",
            channel="memory_eval",
        )
        for item in sample.pending_deltas:
            await memory.apply_memory_delta(
                req,
                MemoryDelta(
                    key=str(item["key"]),
                    value=str(item["value"]),
                    source=str(item.get("source") or "user"),
                ),
            )
        summary = await memory.finalize_session(req)
        hot = getattr(memory, "_hot", None)
        kv: dict[str, str] = {}
        if hot is not None:
            kv = user_file_kv(hot.get_raw_user(req.tenant_id, req.user_id))
        score = kv_match_score(kv, sample.expected_kv)
        row.score = score
        row.passed = score >= min_score and summary.get("l4_merged", 0) >= 0
        row.detail = json.dumps(
            {"user_kv": kv, "l4_merged": summary.get("l4_merged")}, ensure_ascii=False
        )
    except Exception as exc:
        row.error = str(exc)
        row.passed = False
    row.latency_ms = (time.perf_counter() - t0) * 1000
    return row


async def run_memory_eval(
    *,
    memory: Any,
    ctx: RunContext,
    dataset_path: str,
    chat_cfg: Optional[ChatAgentConfig] = None,
    sample_limit: Optional[int] = None,
    session_search_min_score: float = 0.5,
    l1_extract_min_score: float = 1.0,
    use_mock_only: bool = False,
    l4_memory_factory: Any = None,
) -> dict[str, Any]:
    cfg = chat_cfg or load_chat_config()
    samples = load_memory_eval_dataset(dataset_path)
    if sample_limit:
        samples = samples[:sample_limit]
    rows: list[MemoryEvalRow] = []
    for sample in samples:
        if sample.kind == "session_search":
            rows.append(
                await eval_session_search_row(
                    memory,
                    sample,
                    min_score=session_search_min_score,
                )
            )
        elif sample.kind == "l1_extract":
            facts: list[dict] = []
            mock = getattr(sample, "mock_extract", None)
            if mock and isinstance(mock, list) and (use_mock_only or ctx.models is None):
                facts = mock
            elif ctx.models is not None and not use_mock_only:
                turns = []
                if sample.transcript:
                    for line in sample.transcript.splitlines():
                        line = line.strip()
                        if line.lower().startswith("user:"):
                            turns.append({"role": "user", "content": line[5:].strip()})
                        elif line.lower().startswith("assistant:"):
                            turns.append(
                                {"role": "assistant", "content": line[10:].strip()}
                            )
                elif sample.seed_turns:
                    turns = sample.seed_turns
                facts = await extract_l1_facts_from_session(ctx, turns, cfg)
            elif mock:
                facts = mock
            else:
                rows.append(
                    MemoryEvalRow(
                        id=sample.id,
                        kind=sample.kind,
                        passed=False,
                        score=0.0,
                        detail="no models and no mock_extract",
                        tags=sample.tags,
                        error="missing_mock",
                    )
                )
                continue
            extracted = {f["key"]: f["value"] for f in facts}
            score = kv_match_score(extracted, sample.expected_kv)
            row = MemoryEvalRow(
                id=sample.id,
                kind=sample.kind,
                passed=score >= l1_extract_min_score,
                score=score,
                detail=json.dumps(extracted, ensure_ascii=False),
                tags=sample.tags,
            )
            rows.append(row)
        elif sample.kind == "hitl_finalize":
            rows.append(
                await eval_hitl_finalize_row(
                    memory, sample, min_score=l1_extract_min_score
                )
            )
        elif sample.kind == "l4_merge":
            if l4_memory_factory is None:
                rows.append(
                    MemoryEvalRow(
                        id=sample.id,
                        kind=sample.kind,
                        passed=False,
                        error="l4_memory_factory required",
                        tags=sample.tags,
                    )
                )
                continue
            l4_mem = await l4_memory_factory(sample)
            try:
                rows.append(
                    await eval_l4_merge_row(
                        l4_mem, sample, min_score=l1_extract_min_score
                    )
                )
            finally:
                adb = getattr(l4_mem, "_archive_db", None)
                if adb is not None:
                    close = getattr(adb, "close", None)
                    if callable(close):
                        maybe = close()
                        if hasattr(maybe, "__await__"):
                            await maybe
    passed = sum(1 for r in rows if r.passed)
    total = len(rows) or 1
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in ("session_search", "l1_extract", "hitl_finalize", "l4_merge"):
        subset = [r for r in rows if r.kind == kind]
        if not subset:
            continue
        by_kind[kind] = {
            "count": len(subset),
            "passed": sum(1 for r in subset if r.passed),
            "avg_score": sum(r.score for r in subset) / len(subset),
        }
    return {
        "total": len(rows),
        "passed": passed,
        "pass_rate": passed / total,
        "by_kind": by_kind,
        "rows": [r.to_dict() for r in rows],
    }


def write_memory_eval_report(result: dict[str, Any], out_dir: str | Path) -> Path:
    write_memory_eval_reports(result, out_dir)
    return Path(out_dir) / "summary.json"
