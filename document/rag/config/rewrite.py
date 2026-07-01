from dataclasses import dataclass


@dataclass(frozen=True)
class RewriteConfig:
    enable_hyde: bool = False
    enable_multi_query: bool = False
    multi_query_count: int = 3
