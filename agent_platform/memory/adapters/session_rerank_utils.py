"""L2 会话检索轻量 rerank（token overlap）。"""

from __future__ import annotations

from typing import List


def rerank_message_dicts(
    query: str,
    messages: List[dict],
    *,
    top_n: int,
    content_key: str = "content",
) -> List[dict]:
    if not messages or top_n <= 0:
        return messages[:top_n] if top_n else messages

    tokens = [t for t in query.lower().split() if t]
    if not tokens:
        return messages[:top_n]

    scored: List[tuple[float, int, dict]] = []
    for idx, msg in enumerate(messages):
        text = str(msg.get(content_key) or "").lower()
        score = sum(1.0 for t in tokens if t in text) / len(tokens)
        scored.append((score, idx, msg))

    scored.sort(key=lambda x: (-x[0], x[1]))
    out: List[dict] = []
    for score, _, msg in scored[:top_n]:
        enriched = dict(msg)
        enriched["rerank_score"] = score
        out.append(enriched)
    return out
