# -*- coding: utf-8 -*-
"""时间衰减模块单元测试。"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from agent_platform.memory.adapters.time_decay import (
    time_decay_factor,
    apply_time_decay_to_fragments,
)


class TestTimeDecayFactor:
    def test_empty_ts_returns_1(self):
        assert time_decay_factor("") == 1.0
        assert time_decay_factor("  ") == 1.0

    def test_future_time_returns_1(self):
        future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        assert time_decay_factor(future) == 1.0

    def test_just_now_near_1(self):
        now = datetime.now(timezone.utc)
        ts = now.isoformat()
        factor = time_decay_factor(ts, now=now)
        assert factor > 0.99

    def test_half_life_exact(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        half_life_ago = now - timedelta(days=90)
        factor = time_decay_factor(
            half_life_ago.isoformat(), now=now, half_life_days=90.0
        )
        assert abs(factor - 0.5) < 0.01

    def test_double_half_life(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        two_half_lives_ago = now - timedelta(days=180)
        factor = time_decay_factor(
            two_half_lives_ago.isoformat(), now=now, half_life_days=90.0
        )
        assert abs(factor - 0.25) < 0.01

    def test_invalid_ts_returns_1(self):
        assert time_decay_factor("not-a-date") == 1.0

    def test_date_only_format(self):
        now = datetime(2026, 6, 1, tzinfo=timezone.utc)
        ts = "2026-01-01"
        factor = time_decay_factor(ts, now=now, half_life_days=90.0)
        assert 0.0 < factor < 1.0

    def test_zero_half_life_returns_1(self):
        ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        assert time_decay_factor(ts, half_life_days=0) == 1.0

    def test_naive_ts_treated_as_utc(self):
        now = datetime(2026, 6, 1, tzinfo=timezone.utc)
        ts = "2026-03-01T00:00:00"  # no tzinfo
        factor = time_decay_factor(ts, now=now)
        assert 0.0 < factor <= 1.0


class TestApplyTimeDecayToFragments:
    def test_empty_list(self):
        assert apply_time_decay_to_fragments([]) == []

    def test_decay_applied(self):
        now = datetime(2026, 6, 1, tzinfo=timezone.utc)
        fragments = [
            {"ts": (now - timedelta(days=10)).isoformat(), "score": 1.0, "text": "recent"},
            {"ts": (now - timedelta(days=180)).isoformat(), "score": 1.0, "text": "old"},
        ]
        result = apply_time_decay_to_fragments(fragments, now=now, half_life_days=90.0)
        assert len(result) == 2
        # recent should have higher decayed score → sorted first
        assert result[0]["text"] == "recent"
        assert result[0]["_decay_factor"] > result[1]["_decay_factor"]

    def test_metadata_fields(self):
        now = datetime.now(timezone.utc)
        fragments = [
            {"ts": now.isoformat(), "score": 0.8},
        ]
        result = apply_time_decay_to_fragments(fragments, now=now)
        assert "_decay_factor" in result[0]
        assert "_original_score" in result[0]
        assert result[0]["_original_score"] == 0.8

    def test_custom_fields(self):
        now = datetime.now(timezone.utc)
        fragments = [
            {"created": now.isoformat(), "relevance": 0.9},
        ]
        result = apply_time_decay_to_fragments(
            fragments, now=now, ts_field="created", score_field="relevance"
        )
        assert "relevance" in result[0]
        assert "_decay_factor" in result[0]
