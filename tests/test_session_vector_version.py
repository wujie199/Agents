from agent_platform.memory.adapters.session_vector_version import (
    compute_session_vector_index_version,
)


def test_compute_session_vector_index_version_mock():
    cfg = {"session_embedding_backend": "mock", "session_embedding_dim": 64}
    assert compute_session_vector_index_version(cfg) == "mock:64"


def test_compute_session_vector_index_version_explicit_override():
    cfg = {
        "session_embedding_backend": "mock",
        "session_vector_index_version": "custom-v2",
    }
    assert compute_session_vector_index_version(cfg) == "custom-v2"
