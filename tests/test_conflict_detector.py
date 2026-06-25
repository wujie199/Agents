# -*- coding: utf-8 -*-
"""L1 冲突检测模块单元测试。"""

from __future__ import annotations

import pytest

from app.agents.memory.conflict_detector import (
    ConflictRecord,
    ConflictStrategy,
    check_l1_write_conflicts,
    compute_confidence,
    is_values_conflicting,
    resolve_conflict,
)


class TestIsValuesConflicting:
    def test_same_value_no_conflict(self):
        assert is_values_conflicting("张三", "张三") is False

    def test_normalized_same(self):
        assert is_values_conflicting("张三", "张三。") is False

    def test_containment_no_conflict(self):
        assert is_values_conflicting("张三", "我叫张三") is False
        assert is_values_conflicting("我叫张三", "张三") is False

    def test_real_conflict(self):
        assert is_values_conflicting("张三", "李四") is True

    def test_empty_no_conflict(self):
        assert is_values_conflicting("", "李四") is False
        assert is_values_conflicting("张三", "") is False

    def test_case_insensitive(self):
        assert is_values_conflicting("Python", "python") is False


class TestComputeConfidence:
    def test_name_intro_high(self):
        conf = compute_confidence("", "我叫张三")
        assert conf >= 0.9

    def test_known_key_high(self):
        conf = compute_confidence("姓名", "张三")
        assert conf >= 0.9

    def test_medium_key(self):
        conf = compute_confidence("项目", "Agent")
        assert 0.7 <= conf < 0.9

    def test_default_medium(self):
        conf = compute_confidence("随便", "值")
        assert 0.5 <= conf < 0.7


class TestResolveConflict:
    def test_no_conflict_overwrite(self):
        record = resolve_conflict("姓名", "张三", "张三")
        assert record.strategy == "overwrite"
        assert record.resolved_value == "张三"
        assert record.needs_hitl is False

    def test_overwrite_strategy(self):
        record = resolve_conflict("姓名", "张三", "李四", strategy="overwrite")
        assert record.resolved_value == "李四"
        assert record.needs_hitl is False

    def test_keep_old_strategy(self):
        record = resolve_conflict("姓名", "张三", "李四", strategy="keep_old")
        assert record.resolved_value == "张三"
        assert record.needs_hitl is False

    def test_ask_user_high_confidence(self):
        record = resolve_conflict(
            "姓名", "张三", "李四",
            strategy="ask_user",
            new_confidence=0.95,
            l1_auto_write_confidence_min=0.9,
        )
        assert record.resolved_value == "李四"
        assert record.needs_hitl is True

    def test_ask_user_low_confidence(self):
        record = resolve_conflict(
            "姓名", "张三", "李四",
            strategy="ask_user",
            new_confidence=0.5,
            l1_auto_write_confidence_min=0.9,
        )
        assert record.resolved_value == "张三"
        assert record.needs_hitl is True


class TestCheckL1WriteConflicts:
    def test_no_existing_facts(self):
        records = check_l1_write_conflicts(
            {}, [{"key": "姓名", "value": "张三"}]
        )
        assert len(records) == 1
        assert records[0].resolved_value == "张三"

    def test_conflict_detected(self):
        records = check_l1_write_conflicts(
            {"姓名": "张三"}, [{"key": "姓名", "value": "李四"}]
        )
        assert len(records) == 1
        assert is_values_conflicting(records[0].old_value, records[0].new_value)

    def test_no_conflict_new_key(self):
        records = check_l1_write_conflicts(
            {"姓名": "张三"}, [{"key": "职业", "value": "工程师"}]
        )
        assert len(records) == 1
        assert records[0].strategy == "overwrite"

    def test_empty_delta_skipped(self):
        records = check_l1_write_conflicts(
            {}, [{"key": "", "value": "x"}, {"key": "k", "value": ""}]
        )
        assert len(records) == 0

    def test_batch_multiple(self):
        records = check_l1_write_conflicts(
            {"姓名": "张三", "职业": "工程师"},
            [
                {"key": "姓名", "value": "李四"},
                {"key": "职业", "value": "设计师"},
                {"key": "语言", "value": "Python"},
            ],
        )
        assert len(records) == 3
