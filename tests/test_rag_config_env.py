"""RAG / 模型配置环境变量覆盖。"""

from agent_platform.model.registry import ModelRegistry
from document.rag.config import load_rag_pipeline_config


def test_rag_env_overrides(monkeypatch, tmp_path):
    cfg_file = tmp_path / "rag_pipeline.yml"
    cfg_file.write_text(
        "retrieval:\n  use_mock_rerank_fallback: true\n  enable_router: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_USE_MOCK_RERANK_FALLBACK", "false")
    monkeypatch.setenv("RAG_ENABLE_ROUTER", "true")
    cfg = load_rag_pipeline_config(config_path=str(cfg_file))
    assert cfg.retrieval.use_mock_rerank_fallback is False
    assert cfg.retrieval.enable_router is True


def test_model_registry_env_overrides(monkeypatch, tmp_path):
    models_file = tmp_path / "models.yml"
    models_file.write_text(
        """
instances:
  embedding_bge:
    kind: embedding
    provider: local_bge
    model_path: /old/emb
  rerank_bge:
    kind: rerank
    provider: local_bge
    model_path: /old/rerank
  ocr_paddle:
    kind: ocr
    provider: paddleocr
    model_root: /old/ocr
roles:
  embedding:
    instance: embedding_bge
  rerank:
    instance: rerank_bge
  ocr:
    instance: ocr_paddle
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_EMBEDDING_MODEL_PATH", "/new/emb")
    monkeypatch.setenv("RAG_RERANK_MODEL_PATH", "/new/rerank")
    monkeypatch.setenv("OCR_MODEL_ROOT", "/new/ocr")
    reg = ModelRegistry(config_path=str(models_file))
    assert reg._get_profile_for_role("embedding").model_path == "/new/emb"
    assert reg._get_profile_for_role("rerank").model_path == "/new/rerank"
    assert reg.get_ocr_model_root() == "/new/ocr"
