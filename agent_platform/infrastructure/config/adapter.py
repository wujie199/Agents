from typing import Any, TypeVar, Type, Optional
from pathlib import Path
import yaml
import os


T = TypeVar("T")


class ConfigPortAdapter:
    def __init__(self, config_dir: Optional[Path] = None):
        self._config_dir = config_dir or Path("config")
        self._cache: dict[str, dict] = {}
    
    @property
    def config_dir(self) -> Path:
        return self._config_dir
    
    def load(self, config_name: str) -> dict:
        if config_name in self._cache:
            return self._cache[config_name]
        
        config_path = self._config_dir / f"{config_name}.yml"
        if not config_path.exists():
            config_path = self._config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            return {}
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        self._cache[config_name] = data
        return data
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        config_name = keys[0]
        config = self.load(config_name)
        
        value = config
        for k in keys[1:]:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def get_typed(self, config_name: str, schema: Type[T]) -> T:
        data = self.load(config_name)
        if hasattr(schema, "from_dict"):
            return schema.from_dict(data)
        return schema(**data)
    
    def reload(self, config_name: str) -> dict:
        if config_name in self._cache:
            del self._cache[config_name]
        return self.load(config_name)
    
    def get_with_env_override(self, config_name: str, key: str, env_var: str, default: Any = None) -> Any:
        env_value = os.environ.get(env_var)
        if env_value is not None:
            return env_value
        return self.get(f"{config_name}.{key}", default)
