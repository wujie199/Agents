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
        from document.rag.components.metadata.rule_keyword import (
            RuleKeywordMetadataEnricher,
            parse_tagging_rules_raw,
        )

        if meta.rules:
            rules, ext_tags = parse_tagging_rules_raw(
                {"rules": meta.rules, "extension_tags": meta.extension_tags or {}}
            )
            return RuleKeywordMetadataEnricher(
                rules=rules,
                extension_tags=ext_tags,
                max_tags=meta.max_tags,
                tag_filename=meta.tag_filename,
            )
        if meta.rules_path:
            return RuleKeywordMetadataEnricher(
                rules_path=meta.rules_path,
                max_tags=meta.max_tags,
                tag_filename=meta.tag_filename,
            )
        legacy_path = Path(config_dir) / "metadata_tagging.yml"
        if legacy_path.is_file():
            return RuleKeywordMetadataEnricher(
                rules_path=str(legacy_path),
                max_tags=meta.max_tags,
                tag_filename=meta.tag_filename,
            )
        raise ValueError(
            "metadata.rules 未配置：请在 config/rag.yml 的 metadata.rules 中定义打标规则"
        )

    raise ValueError(f"未知 metadata backend: {backend!r}")
