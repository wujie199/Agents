"""RAG 配置环境变量覆盖。"""

from document.rag.config import load_rag_pipeline_config


def test_rag_env_overrides(monkeypatch, tmp_path):
    cfg_file = tmp_path / "rag.yml"
    cfg_file.write_text(
        "embedding:\n  model_path: /old/emb\n"
        "rerank:\n  model_path: /old/rerank\n"
        "retrieval:\n  use_mock_rerank_fallback: true\n  enable_router: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL_PATH", "/new/emb")
    monkeypatch.setenv("RAG_RERANK_MODEL_PATH", "/new/rerank")
    monkeypatch.setenv("RAG_USE_MOCK_RERANK_FALLBACK", "false")
    monkeypatch.setenv("RAG_ENABLE_ROUTER", "true")
    cfg = load_rag_pipeline_config(config_path=str(cfg_file))
    assert cfg.embedding.model_path == "/new/emb"
    assert cfg.rerank.model_path == "/new/rerank"
    assert cfg.retrieval.use_mock_rerank_fallback is False
    assert cfg.retrieval.enable_router is True
