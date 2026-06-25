# -*- coding: utf-8 -*-
"""DeepAgent 路由门控单元测试。"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.runtime.adapters.deepagents.routing_gate import (
    should_use_deep_agent,
    _CHITCHAT_RE,
)
from app.runtime.adapters.deepagents.config import DeepAgentConfig
from app.agents.orchestration.chat_config import ChatAgentConfig


def _chat_cfg(**kw) -> ChatAgentConfig:
    return MagicMock(spec=ChatAgentConfig, **kw)


def _da_config(**kw) -> DeepAgentConfig:
    """构建真实的 DeepAgentConfig（避免 MagicMock 迭代问题）。"""
    defaults = dict(
        enable_deep_agent=True,
        deep_agent_semantic_gate=True,
        deep_agent_gate_threshold=0.7,
    )
    defaults.update(kw)
    # 用真实 dataclass 而非 MagicMock，确保 complex_task_patterns 可迭代
    patterns = defaults.pop("complex_task_patterns", None)
    if patterns is not None:
        return DeepAgentConfig(
            enable_deep_agent=defaults["enable_deep_agent"],
            deep_agent_semantic_gate=defaults["deep_agent_semantic_gate"],
            deep_agent_gate_threshold=defaults["deep_agent_gate_threshold"],
            complex_task_patterns=tuple(patterns),
        )
    return DeepAgentConfig(
        enable_deep_agent=defaults["enable_deep_agent"],
        deep_agent_semantic_gate=defaults["deep_agent_semantic_gate"],
        deep_agent_gate_threshold=defaults["deep_agent_gate_threshold"],
    )


class TestChitchatRe:
    def test_chitchat_matched(self):
        assert _CHITCHAT_RE.match("你好")
        assert _CHITCHAT_RE.match("hello")
        assert _CHITCHAT_RE.match("再见")

    def test_not_chitchat(self):
        assert _CHITCHAT_RE.match("帮我写一份报告") is None


class TestShouldUseDeepAgent:
    @patch("app.runtime.adapters.deepagents.routing_gate.is_deep_agents_available", return_value=True)
    def test_disabled_config(self, _mock):
        cfg = _chat_cfg()
        da = _da_config(enable_deep_agent=False)
        assert should_use_deep_agent("帮我写一份报告", cfg, da) is False

    @patch("app.runtime.adapters.deepagents.routing_gate.is_deep_agents_available", return_value=False)
    def test_not_available(self, _mock):
        cfg = _chat_cfg()
        da = _da_config(enable_deep_agent=True)
        assert should_use_deep_agent("帮我写一份报告", cfg, da) is False

    @patch("app.runtime.adapters.deepagents.routing_gate.is_deep_agents_available", return_value=True)
    def test_chitchat_excluded(self, _mock):
        cfg = _chat_cfg()
        da = _da_config()
        assert should_use_deep_agent("你好", cfg, da) is False

    @patch("app.runtime.adapters.deepagents.routing_gate.is_deep_agents_available", return_value=True)
    def test_short_query_excluded(self, _mock):
        cfg = _chat_cfg()
        da = _da_config()
        assert should_use_deep_agent("什么", cfg, da) is False

    @patch("app.runtime.adapters.deepagents.routing_gate.is_deep_agents_available", return_value=True)
    def test_complex_pattern_matched(self, _mock):
        cfg = _chat_cfg()
        da = _da_config()
        assert should_use_deep_agent("请先调研市场再写报告然后提供建议", cfg, da) is True

    @patch("app.runtime.adapters.deepagents.routing_gate.is_deep_agents_available", return_value=True)
    def test_report_pattern_matched(self, _mock):
        cfg = _chat_cfg()
        da = _da_config()
        assert should_use_deep_agent("帮我写一份市场分析报告包含详细数据", cfg, da) is True

    @patch("app.runtime.adapters.deepagents.routing_gate.is_deep_agents_available", return_value=True)
    def test_normal_query_not_matched(self, _mock):
        cfg = _chat_cfg()
        da = _da_config()
        assert should_use_deep_agent("今天天气怎么样？", cfg, da) is False

    @patch("app.runtime.adapters.deepagents.routing_gate.is_deep_agents_available", return_value=True)
    def test_empty_query(self, _mock):
        cfg = _chat_cfg()
        da = _da_config()
        assert should_use_deep_agent("", cfg, da) is False
