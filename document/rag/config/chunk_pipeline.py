"""七步分块流水线配置。"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


DOMAIN_GRANULARITY: Dict[str, Dict[str, int]] = {
    "legal": {"target_min": 100, "target_max": 350, "target_ideal": 200},
    "medical": {"target_min": 128, "target_max": 384, "target_ideal": 220},
    "general": {"target_min": 200, "target_max": 512, "target_ideal": 300},
    "narrative": {"target_min": 256, "target_max": 768, "target_ideal": 400},
    "faq": {"target_min": 0, "target_max": 0, "target_ideal": 0},
    "code": {"target_min": 0, "target_max": 0, "target_ideal": 0},
}


@dataclass(frozen=True)
class ChunkPipelineConfig:
    """七步分块流水线参数（Step1–Step7）。"""

    enabled: bool = True
    domain: str = "general"
    min_unit_size: int = 30
    min_chunk_size: int = 30
    max_chunk_size: int = 1500

    # Step2 语义边界
    topic_tiling_window: int = 2
    topic_tiling_threshold: float = 0.30
    embedding_drop_threshold: float = 0.15
    use_embedding_boundary: bool = True
    min_sentence_chars: int = 4

    # Step3 粒度（可被 domain 覆盖）
    target_min: int = 200
    target_max: int = 512
    target_ideal: int = 300
    density_high_threshold: float = 0.7
    density_low_threshold: float = 0.3

    # Step4 父子层级
    enable_parent_child: bool = True
    parent_target_chars: int = 800
    child_target_chars: int = 200
    enable_contextual_prefix: bool = True
    contextual_prefix_max_chars: int = 200

    # Step5 质量评分
    quality_hard_filter: float = 0.3
    quality_soft_filter: float = 0.5
    quality_repair_threshold: float = 0.8

    # Step6 上下文修复
    enable_context_repair: bool = True
    entity_definition_max_chars: int = 50

    # Step7 去重
    enable_exact_dedupe: bool = True
    enable_semantic_dedupe: bool = True
    semantic_dedup_threshold: float = 0.92
    overlap_ratio: float = 0.0
    parent_store_dir: str = "parent_chunks"

    # OCR / 格式
    preserve_tables: bool = True
    preserve_code_blocks: bool = True
    preserve_faq_pairs: bool = True

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "ChunkPipelineConfig":
        if not raw:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known and k != "extra"}
        extra = {k: v for k, v in raw.items() if k not in known}
        cfg = cls(**kwargs)
        if extra:
            object.__setattr__(cfg, "extra", extra)
        return cfg

    def with_domain(self, domain: Optional[str] = None) -> "ChunkPipelineConfig":
        key = (domain or self.domain or "general").lower()
        gran = DOMAIN_GRANULARITY.get(key)
        if not gran:
            return self
        return ChunkPipelineConfig(
            **{
                **{f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()},  # type: ignore[attr-defined]
                "domain": key,
                **gran,
            }
        )


def parse_chunk_pipeline_config(raw: Optional[Dict[str, Any]]) -> ChunkPipelineConfig:
    cfg = ChunkPipelineConfig.from_dict(raw or {})
    return cfg.with_domain(cfg.domain)
