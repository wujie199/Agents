"""model_bridge 消息过滤。"""

from agent_platform.model.messages import filter_dict_messages_for_llm
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import RemoveMessage

from app.runtime.adapters.langgraph.model_bridge import (
    _message_to_dict,
    filter_messages_for_llm,
)


def test_filter_messages_for_llm_skips_remove():
    msgs = [
        RemoveMessage(id="x"),
        HumanMessage(content="hi"),
        AIMessage(content="ok"),
    ]
    filtered = filter_messages_for_llm(msgs)
    assert len(filtered) == 2
    assert all(_message_to_dict(m) is not None for m in filtered)
    assert _message_to_dict(RemoveMessage(id="y")) is None


def test_filter_dict_messages_integration_with_remove_dict():
    lc_filtered = [
        _message_to_dict(m)
        for m in filter_messages_for_llm(
            [RemoveMessage(id="x"), HumanMessage(content="hi")]
        )
    ]
    lc_filtered = [d for d in lc_filtered if d]
    assert filter_dict_messages_for_llm(
        [{"role": "remove", "content": ""}, *lc_filtered]
    ) == [{"role": "user", "content": "hi"}]
