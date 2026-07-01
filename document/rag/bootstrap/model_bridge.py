"""RAG 组装辅助：将 models.yml 中的模型路径注入管道配置。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Tuple

from document.rag.config.pipeline import RagPipelineConfig


def ensure_model_registry(
    models: Any,
    *,
    config_dir: str = "config",
) -> Any:
    if models is not None:
        return models
    from agent_platform.model.registry import ModelRegistry

    return ModelRegistry(config_path=f"{config_dir}/models.yml")


def apply_models_to_rag_config(
    cfg: RagPipelineConfig,
    models: Any,
    *,
    config_dir: str = "config",
) -> Tuple[RagPipelineConfig, Any]:
    registry = ensure_model_registry(models, config_dir=config_dir)
    ocr_root = registry.get_ocr_model_root()
    if ocr_root:
        cfg = replace(cfg, ingest=replace(cfg.ingest, ocr_model_root=ocr_root))
    return cfg, registry
