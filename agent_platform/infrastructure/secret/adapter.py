import os
from typing import Optional
import hashlib


class SecretPortAdapter:
    def __init__(self, secrets: Optional[dict] = None):
        self._secrets = secrets or {}
        self._env_prefix = ""
    
    def set_env_prefix(self, prefix: str) -> None:
        self._env_prefix = prefix
    
    def get_secret(self, key: str) -> Optional[str]:
        env_key = f"{self._env_prefix}{key}".upper().replace(".", "_")
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value
        return self._secrets.get(key)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        return self.get_secret(f"{provider}_api_key")
    
    def require_secret(self, key: str) -> str:
        value = self.get_secret(key)
        if value is None:
            raise ValueError(f"Secret not found: {key}")
        return value
    
    def mask_in_logs(self, value: str) -> str:
        if not value:
            return ""
        if len(value) <= 4:
            return "****"
        return value[:2] + "****" + value[-2:]
    
    def hash_secret(self, value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def load_from_config(self, config: dict, key_mappings: dict) -> None:
        for config_key, secret_key in key_mappings.items():
            if config_key in config:
                self._secrets[secret_key] = config[config_key]
