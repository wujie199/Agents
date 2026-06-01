from agent_platform.memory.adapters.session_search_fusion import rrf_merge_messages


def test_rrf_merge_messages_prefers_both_lists():
    fts = [
        {"message_id": "a", "content": "alpha"},
        {"message_id": "b", "content": "beta"},
    ]
    vector = [
        {"message_id": "b", "content": "beta"},
        {"message_id": "c", "content": "gamma"},
    ]
    merged = rrf_merge_messages([fts, vector])
    ids = [m["message_id"] for m in merged]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}
