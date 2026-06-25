"""Prompt 组装：证据 grounding 与滚动摘要。"""

from __future__ import annotations

from core.domain.evidence import Evidence, EvidenceBundle, SourceType

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.context_builder import _format_turns_for_summary
from app.agents.prompts.prompt_builder import (
    EVIDENCE_GROUNDING_RULES,
    format_evidence_bundle,
    build_chat_messages,
)


def test_format_evidence_strict_grounding():
    bundle = EvidenceBundle(
        evidences=[
            Evidence(
                id="1",
                content="米家1C 适合小户型",
                score=0.9,
                source_type=SourceType.VECTOR,
            )
        ],
        empty=False,
    )
    text = format_evidence_bundle(bundle, strict_grounding=True)
    assert EVIDENCE_GROUNDING_RULES in text
    assert "米家1C" in text


def test_rolling_summary_user_only():
    turns = [
        {"role": "user", "content": "品牌有哪些"},
        {"role": "assistant", "content": "小红书/LG 是常见品牌"},
    ]
    all_text = _format_turns_for_summary(turns, user_only=False)
    user_text = _format_turns_for_summary(turns, user_only=True)
    assert "小红书" in all_text
    assert "小红书" not in user_text
    assert "品牌有哪些" in user_text


def test_build_chat_messages_tool_hints():
    msgs = build_chat_messages(
        memory_system="sys",
        user_message="之前说过什么",
        tool_hints="请调用 session_search",
    )
    assert "session_search" in msgs[0]["content"]
