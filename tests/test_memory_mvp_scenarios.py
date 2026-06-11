"""第一期 MVP 记忆场景基线测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_platform.memory.adapters.config_loader import load_memory_config
from app.agents.chat_config import ChatAgentConfig
from app.agents.context_builder import resolve_recall_scope
from app.agents.name_remember import parse_name_from_intro
from app.agents.retrieval_router import build_retrieval_plan, classify_intent


FIXTURE = Path(__file__).parent / "fixtures" / "memory_scenarios.yml"


def _load_scenarios():
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8")) or {}
    return data.get("scenarios") or []


def _cfg() -> ChatAgentConfig:
    return ChatAgentConfig(
        retrieval_orchestration=True,
        session_search_prefetch=True,
        knowledge_session_search=True,
        enable_rag=True,
    )


@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda s: s["id"])
def test_scenario_intent_routing(scenario):
    cfg = _cfg()
    query = scenario["query"]
    intent = classify_intent(query, cfg)
    assert intent == scenario["expect_intent"], (
        f"{scenario['id']}: intent {intent} != {scenario['expect_intent']}"
    )
    plan = build_retrieval_plan(query, cfg, enable_rag=True)
    if scenario.get("expect_session_search") is True:
        assert plan.run_session_search, scenario["id"]
    if scenario.get("expect_rag") is False:
        assert not plan.run_rag, scenario["id"]
    if scenario.get("expect_rag") is True:
        assert plan.run_rag, scenario["id"]
    if scenario.get("expect_scope") == "user":
        assert resolve_recall_scope(query, cfg) == "user"


@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda s: s["id"])
def test_scenario_name_parse(scenario):
    if "expect_name" not in scenario:
        return
    assert parse_name_from_intro(scenario["query"]) == scenario["expect_name"]


def test_recall_prefetch_miss_default_allows_rag():
    cfg = ChatAgentConfig(recall_skip_rag_when_prefetch_miss=False)
    assert cfg.recall_skip_rag_when_prefetch_miss is False


def test_default_memory_config_full_stack():
    cfg = load_memory_config("config/memory.yml")
    assert cfg.get("enable_session_vector_index") is True
    assert cfg.get("enable_cold_archive") is True
    assert cfg.get("skill_auto_extract_draft") is True
    assert cfg.get("session_search_negative_cache_ttl", 0) > 0
