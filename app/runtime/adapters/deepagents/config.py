# -*- coding: utf-8 -*-
"""DeepAgents 配置映射：从 config/chat.yml 的 deep_agent 段读取。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pathlib import Path


@dataclass(frozen=True)
class DeepAgentConfig:
    """DeepAgents 外层规划 harness 配置。"""

    # 总开关
    enable_deep_agent: bool = False
    # LLM 语义门控：是否用 LLM 判断复杂任务
    deep_agent_semantic_gate: bool = True
    # LLM 判断门控阈值（0.0-1.0）
    deep_agent_gate_threshold: float = 0.7
    # 规划 LLM role
    planning_model_role: str = "main_llm"
    # 子Agent 定义（name, description, prompt, tools, model）
    subagents: Tuple[Dict[str, Any], ...] = ()
    # HITL 中断配置：哪些 tool 需要人工审批
    interrupt_on: Optional[Dict[str, Any]] = None
    # 外层规划 system prompt
    system_prompt: str = (
        "你是企业助手。简单问题委托 chat-worker，"
        "研究任务委托 research-agent，报告生成委托 report-writer。"
    )
    # 自定义工具（传给 DeepAgent 的顶层 tools）
    custom_tool_names: Tuple[str, ...] = ()
    # 多步骤指令正则模式
    complex_task_patterns: Tuple[str, ...] = (
        r"先.*再.*然后",
        r"帮我写.*报告",
        r"对比分析",
        r"制定.*方案",
        r"写一份.*文档",
        r"分析.*并.*建议",
    )


_DEFAULT = DeepAgentConfig()


def load_deep_agent_config(
    config_dir: str | Path = "config",
) -> DeepAgentConfig:
    """从 config/chat.yml 的 deep_agent 段读取配置。"""
    path = Path(config_dir) / "chat.yml"
    if not path.is_file():
        return _DEFAULT
    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    da = raw.get("deep_agent") or {}
    if not isinstance(da, dict):
        return _DEFAULT

    subagents_raw = da.get("subagents") or []
    subagents = tuple(
        dict(s) if isinstance(s, dict) else {}
        for s in subagents_raw
        if isinstance(s, dict)
    )

    interrupt_on = da.get("interrupt_on")
    if interrupt_on and not isinstance(interrupt_on, dict):
        interrupt_on = None

    patterns_raw = da.get("complex_task_patterns") or []
    patterns = tuple(str(p) for p in patterns_raw) if patterns_raw else _DEFAULT.complex_task_patterns

    tool_names_raw = da.get("custom_tool_names") or []
    tool_names = tuple(str(t) for t in tool_names_raw)

    return DeepAgentConfig(
        enable_deep_agent=bool(da.get("enable", _DEFAULT.enable_deep_agent)),
        deep_agent_semantic_gate=bool(
            da.get("semantic_gate", _DEFAULT.deep_agent_semantic_gate)
        ),
        deep_agent_gate_threshold=float(
            da.get("gate_threshold", _DEFAULT.deep_agent_gate_threshold)
        ),
        planning_model_role=str(
            da.get("planning_model_role", _DEFAULT.planning_model_role)
        ),
        subagents=subagents,
        interrupt_on=interrupt_on,
        system_prompt=str(da.get("system_prompt", _DEFAULT.system_prompt)),
        custom_tool_names=tool_names,
        complex_task_patterns=patterns,
    )
