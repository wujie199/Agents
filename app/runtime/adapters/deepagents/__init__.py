# -*- coding: utf-8 -*-
"""DeepAgents 适配层：外层规划 harness。

将 deepagents 作为可选依赖，不装则此模块不可用但不影响核心对话功能。
集成模式：DeepAgents 做外层规划 harness，当前系统做内层执行子Agent。
"""

try:
    import deepagents  # noqa: F401

    _DEEP_AGENTS_AVAILABLE = True
except ImportError:
    _DEEP_AGENTS_AVAILABLE = False


def is_deep_agents_available() -> bool:
    """deepagents 是否已安装。"""
    return _DEEP_AGENTS_AVAILABLE
