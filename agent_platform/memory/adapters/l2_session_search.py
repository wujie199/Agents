# -*- coding: utf-8 -*-
"""Hermes L2 session_search 四形态：Discovery / Scroll / Read / Browse（零 LLM）。"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Awaitable, Dict, List, Literal, Optional, Sequence

SessionSearchMode = Literal["discovery", "scroll", "read", "browse"]

_DISCOVER_SCAN_LIMIT = 300
_CRON_SOURCES = frozenset({"cron", "subagent", "tool"})
_SESSION_LINK_RE = re.compile(
    r"@session:(?:(?P<profile>[^/]+)/)?(?P<session_id>[^/\s]+)",
    re.IGNORECASE,
)


def parse_session_link(text: str) -> Optional[dict[str, str]]:
    """解析 @session:<profile>/<id> 或 @session:<id>。"""
    m = _SESSION_LINK_RE.search(text or "")
    if not m:
        return None
    out: dict[str, str] = {"session_id": m.group("session_id").strip()}
    profile = m.group("profile")
    if profile:
        out["profile"] = profile.strip()
    return out


def format_session_link(session_id: str, profile: str = "") -> str:
    if profile:
        return f"@session:{profile}/{session_id}"
    return f"@session:{session_id}"


async def resolve_lineage_root_async(
    session_id: str,
    parent_lookup: Callable[[str], Awaitable[Optional[str]]],
) -> str:
    current = session_id
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        parent = await parent_lookup(current)
        if not parent:
            return current
        current = parent
    return session_id


async def dedupe_hits_by_lineage_async(
    hits: Sequence[dict],
    *,
    parent_lookup: Callable[[str], Awaitable[Optional[str]]],
    skip_lineage_root: Optional[str] = None,
) -> List[dict]:
    best: dict[str, dict] = {}
    for row in hits:
        sid = str(row.get("session_id") or "")
        if not sid:
            continue
        root = await resolve_lineage_root_async(sid, parent_lookup)
        if skip_lineage_root and root == skip_lineage_root:
            continue
        score = float(row.get("score") or row.get("bm25_score") or 0.0)
        prev = best.get(root)
        if prev is None or score > float(prev.get("score") or 0.0):
            best[root] = {**row, "lineage_root": root, "score": score}
    return sorted(
        best.values(),
        key=lambda r: (-float(r.get("score") or 0.0), str(r.get("ts") or "")),
    )


def apply_cron_downrank(hits: Sequence[dict], *, factor: float = 0.35) -> List[dict]:
    out: List[dict] = []
    for row in hits:
        item = dict(row)
        source = str(item.get("source") or item.get("session_source") or "user")
        score = float(item.get("score") or 0.0)
        if source in _CRON_SOURCES:
            score *= factor
        item["score"] = score
        out.append(item)
    return out


def _preview(content: str, limit: int = 200) -> str:
    text = (content or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def build_bookends_narrative(
    session: dict,
    first_msg: Optional[dict],
    hit_msg: dict,
    last_msg: Optional[dict],
    *,
    profile: str = "",
) -> str:
    sid = str(session.get("session_id") or hit_msg.get("session_id") or "")
    title = str(session.get("title") or sid)
    link = format_session_link(sid, profile)
    lines = [
        f"## {title} ({link})",
        f"started: {session.get('started_at', '')}",
    ]
    if first_msg:
        lines.append(
            f"[opening] [{first_msg.get('ts', '')}] "
            f"{first_msg.get('role', 'user')}: {_preview(str(first_msg.get('content') or ''))}"
        )
    lines.append(
        f"[hit] [{hit_msg.get('ts', '')}] "
        f"{hit_msg.get('role', 'user')}: {_preview(str(hit_msg.get('content') or ''))}"
    )
    if last_msg and last_msg.get("message_id") != hit_msg.get("message_id"):
        lines.append(
            f"[closing] [{last_msg.get('ts', '')}] "
            f"{last_msg.get('role', 'user')}: {_preview(str(last_msg.get('content') or ''))}"
        )
    return "\n".join(lines)


def format_message_row(msg: dict) -> dict:
    return {
        "message_id": str(msg.get("message_id") or ""),
        "session_id": str(msg.get("session_id") or ""),
        "role": str(msg.get("role") or "user"),
        "content": str(msg.get("content") or ""),
        "ts": str(msg.get("ts") or ""),
        "score": float(msg.get("score") or 0.0),
    }


async def run_session_search_mode(
    archive_db: Any,
    *,
    mode: SessionSearchMode,
    context: Any,
    query: str = "",
    limit: int = 3,
    sort: str = "newest",
    around_message_id: str = "",
    before: int = 5,
    after: int = 5,
    session_link: str = "",
    browse_message_limit: int = 3,
) -> str:
    tenant_id = context.tenant_id
    user_id = context.user_id
    current_session = context.session_id

    parent_cache: dict[str, Optional[str]] = {}

    async def parent_lookup(session_id: str) -> Optional[str]:
        if session_id in parent_cache:
            return parent_cache[session_id]
        getter = getattr(archive_db, "get_session_parent_id", None)
        if getter is None:
            parent_cache[session_id] = None
            return None
        parent = getter(session_id)
        if hasattr(parent, "__await__"):
            parent = await parent
        parent_cache[session_id] = parent
        return parent

    current_root = await resolve_lineage_root_async(current_session, parent_lookup)

    if mode == "read":
        link = session_link or query
        parsed = parse_session_link(link)
        if not parsed:
            return json.dumps(
                {"mode": "read", "error": "invalid session link; use @session:<id>"},
                ensure_ascii=False,
            )
        sid = parsed["session_id"]
        session = await archive_db.get_session(sid)
        if session is None:
            return json.dumps(
                {"mode": "read", "error": f"session not found: {sid}"},
                ensure_ascii=False,
            )
        if session.get("tenant_id") != tenant_id or session.get("user_id") != user_id:
            return json.dumps(
                {"mode": "read", "error": "session access denied"},
                ensure_ascii=False,
            )
        messages = await archive_db.select_many(
            "messages",
            ["message_id", "session_id", "role", "content", "ts"],
            where={"session_id": sid},
            order_by="ts ASC",
        )
        return json.dumps(
            {
                "mode": "read",
                "session": session,
                "messages": [format_message_row(m) for m in messages],
            },
            ensure_ascii=False,
        )

    if mode == "scroll":
        if not around_message_id:
            return json.dumps(
                {"mode": "scroll", "error": "around_message_id required"},
                ensure_ascii=False,
            )
        anchor = await archive_db.get_message(around_message_id)
        if anchor is None:
            return json.dumps(
                {"mode": "scroll", "error": f"message not found: {around_message_id}"},
                ensure_ascii=False,
            )
        sid = str(anchor.get("session_id") or "")
        root = await resolve_lineage_root_async(sid, parent_lookup)
        if root == current_root:
            return json.dumps(
                {
                    "mode": "scroll",
                    "error": "cannot scroll current session lineage (already in context)",
                },
                ensure_ascii=False,
            )
        session = await archive_db.get_session(sid)
        if session is None or session.get("tenant_id") != tenant_id:
            return json.dumps(
                {"mode": "scroll", "error": "session access denied"},
                ensure_ascii=False,
            )
        window = await archive_db.get_messages_around(
            sid, around_message_id, before=before, after=after
        )
        return json.dumps(
            {
                "mode": "scroll",
                "session_id": sid,
                "lineage_root": root,
                "anchor_message_id": around_message_id,
                "messages": [format_message_row(m) for m in window],
            },
            ensure_ascii=False,
        )

    if mode == "browse":
        sessions = await archive_db.list_sessions_rich(
            tenant_id, user_id, limit=limit, sort=sort
        )
        items: List[dict] = []
        for sess in sessions:
            if sess.get("parent_session_id"):
                continue
            source = str(sess.get("source") or "user")
            if source in _CRON_SOURCES:
                continue
            sid = str(sess.get("session_id") or "")
            msgs = await archive_db.select_many(
                "messages",
                ["message_id", "session_id", "role", "content", "ts"],
                where={"session_id": sid},
                order_by="ts DESC",
                limit=browse_message_limit,
            )
            msgs.reverse()
            items.append(
                {
                    "session": sess,
                    "messages": [format_message_row(m) for m in msgs],
                }
            )
        return json.dumps({"mode": "browse", "sessions": items}, ensure_ascii=False)

    q = (query or "").strip()
    if not q:
        return json.dumps({"mode": "discovery", "error": "query required"}, ensure_ascii=False)

    searcher = getattr(archive_db, "search_messages_ranked", None)
    if searcher is None:
        hits = await archive_db.search_messages(
            user_id=user_id,
            tenant_id=tenant_id,
            query=q,
            limit=_DISCOVER_SCAN_LIMIT,
        )
    else:
        hits = await searcher(
            user_id=user_id,
            tenant_id=tenant_id,
            query=q,
            limit=_DISCOVER_SCAN_LIMIT,
        )

    hits = apply_cron_downrank(hits)
    if sort == "oldest":
        hits = sorted(hits, key=lambda r: str(r.get("ts") or ""))
    deduped = await dedupe_hits_by_lineage_async(
        hits,
        parent_lookup=parent_lookup,
        skip_lineage_root=current_root,
    )
    deduped = deduped[: max(1, min(limit, 10))]

    narratives: List[str] = []
    results: List[dict] = []
    for hit in deduped:
        sid = str(hit.get("session_id") or "")
        session = await archive_db.get_session(sid) or {"session_id": sid}
        first_msg = await archive_db.get_session_bookend(sid, which="first")
        last_msg = await archive_db.get_session_bookend(sid, which="last")
        narrative = build_bookends_narrative(session, first_msg, hit, last_msg)
        narratives.append(narrative)
        results.append(
            {
                "session_id": sid,
                "lineage_root": hit.get("lineage_root"),
                "hit": format_message_row(hit),
                "score": hit.get("score"),
            }
        )

    return json.dumps(
        {
            "mode": "discovery",
            "query": q,
            "count": len(results),
            "results": results,
            "narrative": "\n\n".join(narratives),
        },
        ensure_ascii=False,
    )
