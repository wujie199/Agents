"""LLM 消息 dict 过滤与规范化。"""

import pytest

from agent_platform.model.messages import (
    filter_dict_messages_for_llm,
    normalize_llm_input,
)


def test_filter_dict_messages_for_llm_skips_remove():
    msgs = [
        {"role": "remove", "content": ""},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"not": "a dict role"},
        "bad",
    ]
    filtered = filter_dict_messages_for_llm(msgs)
    assert filtered == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]


def test_normalize_llm_input_from_string():
    assert normalize_llm_input("hello") == [
        {"role": "user", "content": "hello"},
    ]


def test_normalize_llm_input_preserves_messages():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert normalize_llm_input(msgs) == msgs


@pytest.mark.asyncio
async def test_multi_query_expand_uses_messages_format():
    captured = {}

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            captured["messages"] = messages
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Msg",
                                    (),
                                    {"content": "子查询1\n子查询2"},
                                )()
                            },
                        )()
                    ]
                },
            )()

    from document.rag.application.retrieval.rewrite.multi_query import MultiQueryExpander

    expander = MultiQueryExpander(llm_model=MockLLM(), num_queries=2)
    out = await expander.expand("清洁保养扫地机器人")

    assert captured["messages"] == [
        {"role": "user", "content": captured["messages"][0]["content"]},
    ]
    assert "清洁保养扫地机器人" in captured["messages"][0]["content"]
    assert out[0] == "清洁保养扫地机器人"
    assert "子查询1" in out


@pytest.mark.asyncio
async def test_multi_query_expand_fallback_on_api_error():
    class BadLLM:
        async def ainvoke(self, messages, **kwargs):
            raise RuntimeError("400 - messages too short")

    from document.rag.application.retrieval.rewrite.multi_query import MultiQueryExpander

    expander = MultiQueryExpander(llm_model=BadLLM(), num_queries=2)
    out = await expander.expand("原始问题")
    assert out == ["原始问题"]


@pytest.mark.asyncio
async def test_model_wrapper_coerces_string_prompt():
    from agent_platform.model.registry import ModelRegistry, ModelWrapper

    captured = {}

    class FakeProvider:
        async def ainvoke(self, messages, **kwargs):
            captured["messages"] = messages
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Msg",
                                    (),
                                    {"content": "ok"},
                                )()
                            },
                        )()
                    ]
                },
            )()

    reg = ModelRegistry.__new__(ModelRegistry)
    reg._get_provider = lambda _name: FakeProvider()
    reg._get_circuit_breaker = lambda _key: type(
        "CB",
        (),
        {
            "is_open": lambda self: False,
            "record_success": lambda self: None,
            "record_failure": lambda self: None,
        },
    )()

    wrapper = ModelWrapper(
        registry=reg,
        role="router_llm",
        profile_name="test",
        fallback_chain=[],
        resilience={},
    )
    await wrapper.ainvoke("plain prompt")
    assert captured["messages"] == [{"role": "user", "content": "plain prompt"}]
