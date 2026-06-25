# -*- coding: utf-8 -*-
"""路由门控：判断何时走 DeepAgents 规划层，何时直连内层图。

判断条件（可配）：
- 硬规则：多步骤指令正则命中 → 走 DeepAgent
- 硬规则：短句/寒暄/单步查询 → 不走
- 软规则：LLM 语义判断（可选）
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.runtime.adapters.deepagents.config import DeepAgentConfig
from app.runtime.adapters.deepagents import is_deep_agents_available

logger = logging.getLogger(__name__)


# 寒暄 / 短句快速排除
_CHITCHAT_RE = re.compile(
    r"^(你好|您好|hi|hello|hey|在吗|谢谢|感谢|再见|拜拜|"
    r"你叫什么|你是谁|什么模型|who are you)[\s?!.，。~]*$",
    re.I,
)


def should_use_deep_agent(
    user_input: str,
    chat_cfg: ChatAgentConfig,
    da_config: DeepAgentConfig,
) -> bool:
    """判断是否走 DeepAgents 规划层。

    Returns:
        True → 走 DeepAgents harness（复杂任务，需规划+子Agent调度）
        False → 直连内层图（简单问题，prepare → agent → persist）
    """
    # 0. DeepAgent 未启用或未安装 → 不走
    if not da_config.enable_deep_agent:
        return False
    if not is_deep_agents_available():
        return False

    q = (user_input or "").strip()
    if not q:
        return False

    # 1. 硬规则排除：寒暄 / 短句
    if len(q) < 15 or _CHITCHAT_RE.match(q):
        return False

    # 2. 硬规则命中：多步骤指令正则
    for pattern in da_config.complex_task_patterns:
        try:
            if re.search(pattern, q):
                return True
        except re.error:
            continue

    # 3. 软规则：LLM 语义判断（可选，延迟调用避免每轮额外 LLM 开销）
    # 此处返回 False，LLM 判断在 should_use_deep_agent_async 中实现
    return False


async def should_use_deep_agent_async(
    user_input: str,
    chat_cfg: ChatAgentConfig,
    da_config: DeepAgentConfig,
    *,
    models: Any = None,
) -> bool:
    """异步版本：支持 LLM 语义判断门控。"""
    # 先走同步硬规则
    sync_result = should_use_deep_agent(user_input, chat_cfg, da_config)
    if sync_result:
        return True

    # 硬规则排除后，如果不需要语义判断 → 不走
    if not da_config.deep_agent_semantic_gate or models is None:
        return False

    # LLM 语义判断
    q = (user_input or "").strip()
    if len(q) < 15:
        return False

    try:
        llm = models.get_model(chat_cfg.retrieval_router_role)
        prompt = (
            "判断以下用户问题是否需要多步骤规划（如分解任务、写报告、对比分析）。\n"
            "只输出 JSON：{\"complex\": true/false, \"confidence\": 0.0-1.0}\n"
            f"用户问题：{q[:500]}"
        )
        if hasattr(llm, "ainvoke"):
            resp = await llm.ainvoke(prompt)
        else:
            resp = llm.invoke(prompt)
        text = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))

        import json
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(text[start:end+1])
            is_complex = bool(data.get("complex", False))
            confidence = float(data.get("confidence", 0.0))
            return is_complex and confidence >= da_config.deep_agent_gate_threshold
    except Exception as e:
        logger.warning("DeepAgent semantic gate LLM failed: %s", e)

    return False
