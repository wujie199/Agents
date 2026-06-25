from typing import Optional, Any
import yaml
import time
import threading
from pathlib import Path
from core.ports.policy import PolicyResult
from dataclasses import dataclass


@dataclass
class PolicyConfig:
    max_parallel_sends: int = 5
    max_intra_batch_workers: int = 3
    default_batch_size: int = 10
    min_batch_size: int = 1
    max_batch_size: int = 50
    max_rag_batch_queries: int = 10
    max_embed_batch_size: int = 32
    max_qps_per_tenant: int = 100
    token_budget_per_user: int = 100000


class PolicyPortAdapter:
    def __init__(self, config_path: Optional[str] = None):
        self._default_config = PolicyConfig()
        self._tenant_configs: dict[str, PolicyConfig] = {}
        # QPS 限流：tenant → (timestamps)
        self._qps_tracker: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        
        if config_path:
            self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            return
        
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        if "default" in config:
            self._default_config = PolicyConfig(
                max_parallel_sends=config["default"].get("max_parallel_sends", 5),
                max_intra_batch_workers=config["default"].get("max_intra_batch_workers", 3),
                default_batch_size=config["default"].get("default_batch_size", 10),
                min_batch_size=config["default"].get("min_batch_size", 1),
                max_batch_size=config["default"].get("max_batch_size", 50),
                max_rag_batch_queries=config["default"].get("max_rag_batch_queries", 10),
                max_embed_batch_size=config["default"].get("max_embed_batch_size", 32),
            )
        
        tenants = config.get("tenants", {})
        for tenant_id, tenant_config in tenants.items():
            self._tenant_configs[tenant_id] = PolicyConfig(
                max_parallel_sends=tenant_config.get("max_parallel_sends", self._default_config.max_parallel_sends),
                max_intra_batch_workers=tenant_config.get("max_intra_batch_workers", self._default_config.max_intra_batch_workers),
                default_batch_size=tenant_config.get("default_batch_size", self._default_config.default_batch_size),
                min_batch_size=tenant_config.get("min_batch_size", self._default_config.min_batch_size),
                max_batch_size=tenant_config.get("max_batch_size", self._default_config.max_batch_size),
                max_rag_batch_queries=tenant_config.get("max_rag_batch_queries", self._default_config.max_rag_batch_queries),
                max_embed_batch_size=tenant_config.get("max_embed_batch_size", self._default_config.max_embed_batch_size),
            )
    
    def _get_config(self, tenant_id: str) -> PolicyConfig:
        return self._tenant_configs.get(tenant_id, self._default_config)
    
    def check_rate_limit(
        self,
        tenant_id: str,
        user_id: str,
        action: str
    ) -> PolicyResult:
        config = self._get_config(tenant_id)
        if config.max_qps_per_tenant <= 0:
            return PolicyResult(allowed=True)
        now = time.monotonic()
        with self._lock:
            calls = self._qps_tracker.setdefault(tenant_id, [])
            calls[:] = [t for t in calls if now - t < 1.0]
            if len(calls) >= config.max_qps_per_tenant:
                return PolicyResult(
                    allowed=False,
                    reason=f"QPS limit exceeded: {len(calls)}/{config.max_qps_per_tenant}",
                    suggested_value=config.max_qps_per_tenant,
                )
            calls.append(now)
        return PolicyResult(allowed=True)
    
    def check_token_budget(
        self,
        tenant_id: str,
        user_id: str,
        requested_tokens: int
    ) -> PolicyResult:
        config = self._get_config(tenant_id)
        if requested_tokens > config.token_budget_per_user:
            return PolicyResult(
                allowed=False,
                reason=f"Token budget exceeded: {requested_tokens}/{config.token_budget_per_user}",
                suggested_value=config.token_budget_per_user,
            )
        return PolicyResult(allowed=True)
    
    def get_max_parallel_sends(self, tenant_id: str) -> int:
        config = self._get_config(tenant_id)
        return config.max_parallel_sends
    
    def get_max_intra_batch_workers(self, tenant_id: str) -> int:
        config = self._get_config(tenant_id)
        return config.max_intra_batch_workers
    
    def get_batch_size(
        self,
        tenant_id: str,
        context: Optional[dict] = None
    ) -> int:
        config = self._get_config(tenant_id)
        return config.default_batch_size
    
    def suggest_batch_size(
        self,
        tenant_id: str,
        item_count: int,
        context: Optional[dict] = None
    ) -> int:
        config = self._get_config(tenant_id)
        
        if item_count <= 0:
            return config.min_batch_size
        
        suggested = min(item_count, config.max_batch_size)
        suggested = max(suggested, config.min_batch_size)
        
        if context and "system_load" in context:
            load = context["system_load"]
            if load > 0.8:
                suggested = max(config.min_batch_size, suggested // 2)
        
        return suggested
    
    def get_max_rag_batch_queries(self, tenant_id: str) -> int:
        config = self._get_config(tenant_id)
        return config.max_rag_batch_queries
    
    def get_max_embed_batch_size(self, tenant_id: str) -> int:
        config = self._get_config(tenant_id)
        return config.max_embed_batch_size
    
    def validate_batch_size(self, tenant_id: str, batch_size: int) -> int:
        config = self._get_config(tenant_id)
        return max(config.min_batch_size, min(batch_size, config.max_batch_size))
