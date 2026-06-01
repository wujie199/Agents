from typing import Optional, Set, FrozenSet
from core.domain.context import ACL
import yaml
from pathlib import Path


class IdentityPortAdapter:
    def __init__(self, config_path: Optional[str] = None):
        self._config_path = config_path
        self._tenants: dict = {}
        self._users: dict = {}
        self._acl_cache: dict = {}
        
        if config_path:
            self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            return
        
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        self._tenants = config.get("tenants", {})
        self._users = config.get("users", {})
    
    def validate_tenant(self, tenant_id: str) -> bool:
        if not tenant_id:
            return False
        if not self._tenants:
            return True
        return tenant_id in self._tenants
    
    def validate_user(self, tenant_id: str, user_id: str) -> bool:
        if not tenant_id or not user_id:
            return False
        if not self._users:
            return True
        
        tenant_users = self._users.get(tenant_id, {})
        return user_id in tenant_users
    
    def resolve_acl(
        self,
        tenant_id: str,
        user_id: str,
        channel: str
    ) -> ACL:
        cache_key = f"{tenant_id}:{user_id}:{channel}"
        if cache_key in self._acl_cache:
            return self._acl_cache[cache_key]
        
        doc_ids: FrozenSet[str] = frozenset()
        tool_names: FrozenSet[str] = frozenset([
            "read_json_all_title",
            "read_json_context_by_title",
            "save_result_2_json"
        ])
        mcp_servers: FrozenSet[str] = frozenset()
        
        if self._users and tenant_id in self._users:
            user_config = self._users[tenant_id].get(user_id, {})
            if "doc_ids" in user_config:
                doc_ids = frozenset(user_config["doc_ids"])
            if "tools" in user_config:
                tool_names = frozenset(user_config["tools"])
            if "mcp_servers" in user_config:
                mcp_servers = frozenset(user_config["mcp_servers"])
        
        acl = ACL(
            doc_ids=doc_ids,
            tool_names=tool_names,
            mcp_servers=mcp_servers
        )
        
        self._acl_cache[cache_key] = acl
        return acl
    
    def get_tenant_config(self, tenant_id: str) -> dict:
        if not self._tenants:
            return {}
        return self._tenants.get(tenant_id, {})
    
    def get_user_config(self, tenant_id: str, user_id: str) -> dict:
        if not self._users:
            return {}
        tenant_users = self._users.get(tenant_id, {})
        return tenant_users.get(user_id, {})
    
    def clear_cache(self) -> None:
        self._acl_cache.clear()
