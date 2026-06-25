# -*- coding: utf-8 -*-
"""检索意图 LLM 分类（router_llm）；失败时回退正则。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Tuple

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.roles.retrieval_router import Intent, classify_intent

logger = logging.getLogger(__name__)

_VALID = frozenset({"recall", "knowledge", "skill", "profile", "chitchat", "recall_and_knowledge"})

_PROMPT = """你是检索路由助手。根据用户问题选择唯一意图标签。
标签说明：
- recall: 回忆过往对话、之前说过什么
- knowledge: 查知识库/产品/文档/事实
- recall_and_knowledge: 同时需要回忆过往对话和查知识库（如"上次问的那个方案怎么样了"）
- skill: 执行技能/工具/流程
- profile: 用户画像/部门/职位/CRM/LDAP
- chitchat: 寒暄/元对话/无需检索

示例：
- "我们上次讨论的架构方案进展如何" → recall_and_knowledge
- "帮我回忆一下之前说的技术选型" → recall
- "如何配置 Redis 集群" → knowledge
- "你好" → chitchat
- "帮我生成月度报告" → skill
- "我的部门是什么" → profile

只输出 JSON：{{"intent":"...", "confidence":0.0-1.0}}
用户问题：{query}
"""


async def classify_intent_llm(
    query: str,
    cfg: ChatAgentConfig,
    models: Any,
) -> Tuple[Intent, float]:
    from app.agents.roles.retrieval_router import classify_intent as regex_intent

    q = (query or "").strip()
    if not q:
        return "chitchat", 1.0
    try:
        llm = models.get_model(cfg.retrieval_router_role)
        prompt = _PROMPT.format(query=q[:500])
        if hasattr(llm, "ainvoke"):
            resp = await llm.ainvoke(prompt)
        else:
            resp = llm.invoke(prompt)
        text = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
        data = _parse_json(text)
        intent = str(data.get("intent", "")).lower()
        confidence = float(data.get("confidence", 0.0))
        if intent not in _VALID:
            return regex_intent(q, cfg), 0.0
        return intent, max(0.0, min(1.0, confidence))
    except Exception as e:
        logger.warning("LLM retrieval router failed, fallback regex: %s", e)
        return regex_intent(q, cfg), 0.0


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]+\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}
