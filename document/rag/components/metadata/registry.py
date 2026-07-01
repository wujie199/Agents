"""Metadata 组件独立 registry。"""

from pathlib import Path
from typing import Optional

from core.ports.rag.metadata_enricher import MetadataEnricherPort
from document.rag.config.pipeline import RagPipelineConfig


def build_metadata_enricher(
    cfg: RagPipelineConfig,
    *,
    config_dir: str = "config",
) -> MetadataEnricherPort:
    meta = cfg.metadata

    if not meta.enabled:
        from document.rag.components.metadata.none import NoOpMetadataEnricher
        return NoOpMetadataEnricher()

    backend = (meta.backend or "rule_keyword").lower()

    if backend == "none":
        from document.rag.components.metadata.none import NoOpMetadataEnricher
        return NoOpMetadataEnricher()

    if backend == "rule_keyword":
        from document.rag.components.metadata.rule_keyword import RuleKeywordMetadataEnricher
        rules_path = meta.rules_path or str(Path(config_dir) / "metadata_tagging.yml")
        return RuleKeywordMetadataEnricher(
            rules_path=rules_path,
            max_tags=meta.max_tags,
            tag_filename=meta.tag_filename,
        )

    raise ValueError(f"未知 metadata backend: {backend!r}")
