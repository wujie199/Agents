"""text_sanitize 单元测试。"""

from __future__ import annotations

from app.agents.context_builder import is_allowed_l1_value
from app.agents.text_sanitize import (
    sanitize_memory_text_for_chat,
    sanitize_search_fragment_content,
    strip_model_reasoning,
)


def test_strip_model_reasoning():
    raw = "答案\n<think>secret</think>\n结尾"
    assert "secret" not in strip_model_reasoning(raw)
    assert "答案" in strip_model_reasoning(raw)


def test_sanitize_search_fragment_strips_thinking_and_json():
    raw = (
        '<think>x</think>\n'
        '```json\n{"a":1}\n```\n可见正文'
    )
    out = sanitize_search_fragment_content(raw, role="assistant")
    assert "redacted_thinking" not in out
    assert "```" not in out
    assert "可见正文" in out


def test_sanitize_memory_text_drops_json_output_format():
    raw = "# USER PREFERENCES\n\n语言: 中文\n输出格式: JSON\n"
    out = sanitize_memory_text_for_chat(raw)
    assert "输出格式: JSON" not in out
    assert "【回复风格】" in out


def test_is_allowed_l1_value_rejects_auto_json():
    assert not is_allowed_l1_value("输出格式", "JSON")
    assert is_allowed_l1_value("语言", "中文")


def test_sanitize_turn_content_strips_json_for_assistant():
    from app.agents.text_sanitize import sanitize_turn_content

    raw = '前言\n```json\n{"a":1}\n```\n后文'
    out = sanitize_turn_content(raw, role="assistant")
    assert "```" not in out
    assert "后文" in out or "前言" in out
