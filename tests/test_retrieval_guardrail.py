# -*- coding: utf-8 -*-
"""P0：LLM router guardrail + 硬规则 fast path。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.roles.retrieval_router import (
    build_retrieval_plan_async,
    classify_intent_hard_rules,
    is_ambiguous_for_llm_router,
    resolve_intent_async,
    validate_llm_intent,
)


def _cfg(**kw) -> ChatAgentConfig:
    base = ChatAgentConfig(
        retrieval_orchestration=True,
        retrieval_llm_router=True,
        retrieval_router_confidence_min=0.6,
    )
    if not kw:
        return base
    from dataclasses import replace

    return replace(base, **kw)


def test_hard_rule_blocks_skill_override():
    assert classify_intent_hard_rules("用 json_section 技能", _cfg()) == "skill"
    assert not validate_llm_intent("用 json_section 技能", _cfg(), "knowledge")


def test_hard_rule_profile_name():
    assert classify_intent_hard_rules("叫武杰", _cfg()) == "profile"
    assert not validate_llm_intent("叫武杰", _cfg(), "chitchat")


def test_meta_recall_not_ambiguous():
    assert not is_ambiguous_for_llm_router("之前问过什么", _cfg())


def test_mixed_recall_is_ambiguous():
    assert is_ambiguous_for_llm_router("还记得吗", _cfg())


def test_long_knowledge_not_ambiguous():
    assert not is_ambiguous_for_llm_router("扫地机器人激光导航有什么优点", _cfg())


@pytest.mark.asyncio
async def test_llm_skipped_when_not_ambiguous():
    cfg = _cfg(retrieval_llm_router=True)
    models = MagicMock()
    models.get_model = MagicMock()
    intent, source = await resolve_intent_async(
        "扫地机器人有什么优点", cfg, models=models
    )
    assert intent == "knowledge"
    assert source == "regex"
    models.get_model.assert_not_called()


@pytest.mark.asyncio
async def test_llm_guardrail_rejects_invalid_intent():
    cfg = _cfg(retrieval_llm_router=True)

    async def _fake_llm(*_a, **_k):
        return "skill", 0.95

    import app.agents.prompts.retrieval_llm as mod

    orig = mod.classify_intent_llm
    mod.classify_intent_llm = AsyncMock(side_effect=_fake_llm)
    try:
        intent, source = await resolve_intent_async(
            "还记得吗", cfg, models=MagicMock()
        )
        assert intent == "recall"
        assert source.startswith("regex")
    finally:
        mod.classify_intent_llm = orig


@pytest.mark.asyncio
async def test_llm_applied_only_when_ambiguous_and_valid():
    cfg = _cfg(retrieval_llm_router=True)

    async def _fake_llm(*_a, **_k):
        return "knowledge", 0.88

    import app.agents.prompts.retrieval_llm as mod

    orig = mod.classify_intent_llm
    mod.classify_intent_llm = AsyncMock(side_effect=_fake_llm)
    try:
        intent, source = await resolve_intent_async(
            "还记得吗", cfg, models=MagicMock()
        )
        assert intent == "knowledge"
        assert source == "llm:0.88"
        plan = await build_retrieval_plan_async(
            "还记得吗", cfg, enable_rag=True, models=MagicMock(), ctx=None
        )
        assert plan.intent == "knowledge"
        assert plan.intent_source == "llm:0.88"
        assert plan.recall_strategy in ("none", "semantic")
    finally:
        mod.classify_intent_llm = orig
