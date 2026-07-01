from agent_platform.memory.adapters.session_vector_version import (
    compute_session_vector_index_version,
)


def test_compute_session_vector_index_version_from_models(tmp_path):
    models_file = tmp_path / "models.yml"
    models_file.write_text(
        """
instances:
  embedding_bge:
    kind: embedding
    provider: local_bge
    model_path: /models/emb
roles:
  embedding:
    instance: embedding_bge
""",
        encoding="utf-8",
    )
    from agent_platform.model.registry import ModelRegistry

    reg = ModelRegistry(config_path=str(models_file))
    version = compute_session_vector_index_version({}, models=reg)
    assert version == "local_bge:/models/emb:"


def test_compute_session_vector_index_version_explicit_override():
    cfg = {"session_vector_index_version": "custom-v2"}
    assert compute_session_vector_index_version(cfg) == "custom-v2"
