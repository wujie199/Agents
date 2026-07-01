from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MetadataConfig:
    """元数据打标（backend 见 components/metadata/registry）。"""

    enabled: bool = True
    backend: str = "rule_keyword"
    rules_path: Optional[str] = None
    max_tags: int = 32
    tag_filename: bool = True
