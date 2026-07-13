# -*- coding: utf-8 -*-
"""检索路由 golden：intent + recall_strategy + plan 通道。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.context_builder import resolve_recall_scope
from app.agents.roles.retrieval_router import (
    ROUTING_RULES_VERSION,
    build_retrieval_plan,
    classify_intent,
    is_ambiguous_for_llm_router,
    resolve_intent,
)

FIXTURE = Path(__file__).parent / "fixtures" / "routing_golden.yml"


def _load_scenarios():
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8")) or {}
    assert data.get("version") == ROUTING_RULES_VERSION
    return data.get("scenarios") or []


def _cfg() -> ChatAgentConfig:
    return ChatAgentConfig(
        retrieval_orchestration=True,
        session_search_prefetch=True,
        knowledge_session_search=True,
        enable_rag=True,
        meta_recall_recent_list=True,
        recall_rag_fusion=True,
    )


@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda s: s["id"])
def test_routing_golden_plan(scenario):
    if scenario.get("expect_ambiguous") is True and "expect_intent" not in scenario:
        cfg = _cfg()
        assert is_ambiguous_for_llm_router(scenario["query"], cfg)
        return

    cfg = _cfg()
    query = scenario["query"]
    intent, source = resolve_intent(query, cfg)
    assert intent == scenario["expect_intent"], (
        f"{scenario['id']}: intent {intent} != {scenario['expect_intent']}"
    )
    if expected_source := scenario.get("expect_intent_source"):
        assert source == expected_source, scenario["id"]

    plan = build_retrieval_plan(query, cfg, enable_rag=True)
    assert plan.rules_version == ROUTING_RULES_VERSION
    assert plan.intent == scenario["expect_intent"]

    if expected_strategy := scenario.get("expect_recall_strategy"):
        assert plan.recall_strategy == expected_strategy, scenario["id"]

    if scenario.get("expect_session_search") is True:
        assert plan.run_session_search, scenario["id"]
    if scenario.get("expect_rag") is False:
        assert not plan.run_rag, scenario["id"]
    if scenario.get("expect_rag") is True:
        assert plan.run_rag, scenario["id"]
    if scenario.get("expect_scope") == "user":
        assert plan.recall_scope == "user"
        assert resolve_recall_scope(query, cfg) == "user"


def test_routing_golden_count():
    scenarios = _load_scenarios()
    assert len(scenarios) >= 50


def test_classify_intent_matches_resolve_intent_for_non_hard():
    cfg = _cfg()
    query = "扫地机器人有什么优点"
    assert resolve_intent(query, cfg) == ("knowledge", "regex")
    assert classify_intent(query, cfg) == "knowledge"
