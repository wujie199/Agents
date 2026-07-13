# -*- coding: utf-8
"""Golden dataset for memory evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

MemoryEvalKind = Literal[
    "session_search", "l1_extract", "hitl_finalize", "l4_merge"
]

_VALID_KINDS = frozenset(
    {"session_search", "l1_extract", "hitl_finalize", "l4_merge"}
)


@dataclass
class MemoryEvalSample:
    id: str
    kind: MemoryEvalKind
    tenant_id: str = "eval_tenant"
    user_id: str = "eval_user"
    session_id: str = "eval_session"
    query: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    expected_kv: dict[str, str] = field(default_factory=dict)
    seed_turns: list[dict[str, str]] = field(default_factory=list)
    transcript: str = ""
    mock_extract: list[dict[str, str]] = field(default_factory=list)
    pending_deltas: list[dict[str, str]] = field(default_factory=list)
    l4_facts: list[dict[str, str]] = field(default_factory=list)
    confirm_before_finalize: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": self.query,
            "expected_keywords": list(self.expected_keywords),
            "expected_kv": dict(self.expected_kv),
            "seed_turns": list(self.seed_turns),
            "transcript": self.transcript,
            "mock_extract": list(self.mock_extract),
            "pending_deltas": list(self.pending_deltas),
            "l4_facts": list(self.l4_facts),
            "confirm_before_finalize": self.confirm_before_finalize,
            "tags": list(self.tags),
        }


class DatasetValidationError(ValueError):
    pass


def _parse_row(raw: dict[str, Any], line_no: int) -> MemoryEvalSample:
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"line {line_no}: row must be object")
    sample_id = str(raw.get("id") or "").strip()
    kind = str(raw.get("kind") or "session_search").strip()
    if kind not in _VALID_KINDS:
        raise DatasetValidationError(f"line {line_no}: invalid kind {kind!r}")
    if not sample_id:
        raise DatasetValidationError(f"line {line_no}: missing id")
    return MemoryEvalSample(
        id=sample_id,
        kind=kind,  # type: ignore[arg-type]
        tenant_id=str(raw.get("tenant_id") or "eval_tenant"),
        user_id=str(raw.get("user_id") or "eval_user"),
        session_id=str(raw.get("session_id") or f"eval_{sample_id}"),
        query=str(raw.get("query") or ""),
        expected_keywords=[
            str(k) for k in (raw.get("expected_keywords") or []) if str(k).strip()
        ],
        expected_kv={
            str(k): str(v)
            for k, v in (raw.get("expected_kv") or {}).items()
            if str(k).strip()
        },
        seed_turns=[
            {"role": str(t.get("role")), "content": str(t.get("content") or "")}
            for t in (raw.get("seed_turns") or [])
            if isinstance(t, dict)
        ],
        transcript=str(raw.get("transcript") or ""),
        mock_extract=[
            {"key": str(x.get("key")), "value": str(x.get("value"))}
            for x in (raw.get("mock_extract") or [])
            if isinstance(x, dict) and x.get("key")
        ],
        pending_deltas=[
            {
                "key": str(x.get("key")),
                "value": str(x.get("value")),
                "source": str(x.get("source") or "user"),
            }
            for x in (raw.get("pending_deltas") or [])
            if isinstance(x, dict) and x.get("key")
        ],
        l4_facts=[
            {
                "key": str(x.get("key")),
                "value": str(x.get("value")),
                "source": str(x.get("source") or "ldap"),
            }
            for x in (raw.get("l4_facts") or [])
            if isinstance(x, dict) and x.get("key")
        ],
        confirm_before_finalize=bool(raw.get("confirm_before_finalize", True)),
        tags=[str(t) for t in (raw.get("tags") or [])],
    )


def load_memory_eval_dataset(path: str | Path) -> list[MemoryEvalSample]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    samples: list[MemoryEvalSample] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        samples.append(_parse_row(raw, i))
    return samples


def iter_samples(path: str | Path) -> Iterator[MemoryEvalSample]:
    yield from load_memory_eval_dataset(path)
