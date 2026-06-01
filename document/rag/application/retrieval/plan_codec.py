from typing import Any, Dict, Optional

from document.rag.config import RagPipelineConfig


def business_plan_to_router_override(
    plan: Optional[Any],
    config: Optional[RagPipelineConfig] = None,
) -> Optional[Dict[str, Any]]:
    """
    çåç¬éâ²æ¶ planéå§AGPort é¨?dict / RetrievalPlan dataclasséå¤æµæ¶?RetrievalRouter é¨?plan_overrideé?    """
    if plan is None:
        return None

    if isinstance(plan, dict):
        raw = dict(plan)
    elif hasattr(plan, "__dataclass_fields__"):
        raw = {
            "primary": getattr(plan, "primary_backend", None) or getattr(plan, "primary", None),
            "secondary": getattr(plan, "secondary_backends", None) or getattr(plan, "secondary", []),
            "fusion": getattr(plan, "fusion_strategy", None) or getattr(plan, "fusion", None),
            "cache_policy": getattr(plan, "cache_policy", None),
            "top_k": getattr(plan, "top_k", None),
            "rerank_top_n": getattr(plan, "rerank_top_n", None),
            "graph_hop": getattr(plan, "graph_hop", None),
            "order": getattr(plan, "order", None),
        }
    else:
        return None

    if config:
        raw.setdefault("top_k", config.default_top_k)
        raw.setdefault("rerank_top_n", config.rerank_top_n)
        if "primary" not in raw and "primary_backend" not in raw:
            raw["primary"] = config.retrieval.primary_backend

    if "primary_backend" in raw and "primary" not in raw:
        raw["primary"] = raw.pop("primary_backend")
    if "secondary_backends" in raw and "secondary" not in raw:
        raw["secondary"] = raw.pop("secondary_backends")
    if "fusion_strategy" in raw and "fusion" not in raw:
        raw["fusion"] = raw.pop("fusion_strategy")

    return {k: v for k, v in raw.items() if v is not None}
