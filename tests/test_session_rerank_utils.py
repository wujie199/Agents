import pytest

from agent_platform.memory.adapters.session_rerank_utils import rerank_message_dicts


def test_rerank_orders_by_relevance():
    docs = [
        {"content": "unrelated topic"},
        {"content": "Phoenix project architecture review"},
        {"content": "Phoenix deployment notes"},
    ]
    out = rerank_message_dicts("Phoenix architecture", docs, top_n=2)
    assert len(out) == 2
    assert "Phoenix" in out[0]["content"]
    assert out[0]["rerank_score"] >= out[1]["rerank_score"]
