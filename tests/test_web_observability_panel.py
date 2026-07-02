# -*- coding: utf-8 -*-
"""Web observability panel tests."""

from app.web.observability_panel import format_observability_markdown


def test_format_observability_markdown_full():
    md = format_observability_markdown(
        {
            "trace_id": "web-chat1",
            "node_timing_summary": {
                "total_ms": 1200,
                "nodes": [
                    {"node": "prepare", "duration_ms": 400, "pct": 33.3},
                    {"node": "agent", "duration_ms": 700, "pct": 58.3},
                    {"node": "persist", "duration_ms": 100, "pct": 8.3},
                ],
                "slow_nodes": [],
            },
            "tool_events": [
                {"type": "tool", "tool_name": "session_search", "duration_ms": 45, "error": False},
            ],
            "turn_audit": [
                {
                    "node": "prepare",
                    "duration_ms": 400,
                    "error": None,
                    "content_hashes": {"user_input": "abc123def456"},
                },
            ],
        }
    )
    assert "运行监控" in md
    assert "prepare" in md
    assert "session_search" in md
    assert "审计" in md
