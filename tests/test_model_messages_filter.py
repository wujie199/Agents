"""LLM 消息 dict 过滤。"""

from agent_platform.model.messages import filter_dict_messages_for_llm


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
