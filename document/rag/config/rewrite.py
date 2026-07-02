from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

LlmRewriteMode = str  # never | on_miss | always


@dataclass(frozen=True)
class RewriteProfilePolicy:
    """单类 query 的改写策略。"""

    name: str
    rule: bool = True
    llm: LlmRewriteMode = "never"
    max_hybrid_queries: int = 2


@dataclass(frozen=True)
class TwoStageConfig:
    enabled: bool = True
    top1_min_rerank: float = 0.75
    require_maintenance_source: bool = True


DEFAULT_REWRITE_PROFILES: Dict[str, RewriteProfilePolicy] = {
    "maintenance": RewriteProfilePolicy(
        name="maintenance", rule=True, llm="on_miss", max_hybrid_queries=6
    ),
    "faq_like": RewriteProfilePolicy(
        name="faq_like", rule=True, llm="never", max_hybrid_queries=3
    ),
    "how_to": RewriteProfilePolicy(
        name="how_to", rule=True, llm="on_miss", max_hybrid_queries=4
    ),
    "product_compare": RewriteProfilePolicy(
        name="product_compare", rule=True, llm="on_miss", max_hybrid_queries=4
    ),
    "exact_lookup": RewriteProfilePolicy(
        name="exact_lookup", rule=False, llm="never", max_hybrid_queries=1
    ),
    "generic_knowledge": RewriteProfilePolicy(
        name="generic_knowledge", rule=False, llm="never", max_hybrid_queries=2
    ),
}


@dataclass(frozen=True)
class RewriteConfig:
    enable_hyde: bool = False
    enable_multi_query: bool = True
    multi_query_count: int = 3
    enable_rule_rewrite: bool = True
    rule_max_queries: int = 4
    maintenance_source_boost: float = 0.12
    maintenance_post_rerank_boost: float = 0.18
    faq_non_maintenance_penalty: float = 0.12
    llm_rewrite_once: bool = True
    max_hybrid_queries: int = 6
    hybrid_search_concurrency: int = 4
    default_profile: str = "generic_knowledge"
    profiles: Dict[str, RewriteProfilePolicy] = field(
        default_factory=lambda: dict(DEFAULT_REWRITE_PROFILES)
    )
    two_stage: TwoStageConfig = field(default_factory=TwoStageConfig)

    def profile_policy(self, profile: str) -> RewriteProfilePolicy:
        return self.profiles.get(profile) or self.profiles.get(
            self.default_profile,
            DEFAULT_REWRITE_PROFILES["generic_knowledge"],
        )

    def needs_llm_capability(self) -> bool:
        if self.enable_hyde:
            return True
        if not self.enable_multi_query:
            return False
        return any(p.llm in ("on_miss", "always") for p in self.profiles.values())


def parse_rewrite_profiles(raw: Optional[Mapping]) -> Dict[str, RewriteProfilePolicy]:
    """从 rag.yml rewrite.profiles 解析；缺省项回退 DEFAULT。"""
    if not raw:
        return dict(DEFAULT_REWRITE_PROFILES)
    out = dict(DEFAULT_REWRITE_PROFILES)
    for name, item in raw.items():
        if not isinstance(item, dict):
            continue
        base = out.get(name, RewriteProfilePolicy(name=str(name)))
        llm = str(item.get("llm", base.llm)).lower()
        if llm not in ("never", "on_miss", "always"):
            llm = base.llm
        out[str(name)] = RewriteProfilePolicy(
            name=str(name),
            rule=bool(item.get("rule", base.rule)),
            llm=llm,
            max_hybrid_queries=int(
                item.get("max_hybrid_queries", base.max_hybrid_queries)
            ),
        )
    return out
