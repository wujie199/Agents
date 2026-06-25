from typing import Protocol, Any, TypeVar, Type
from pathlib import Path


T = TypeVar("T")


class ConfigPort(Protocol):
    def load(self, config_name: str) -> dict:
        ...

    def get(self, key: str, default: Any = None) -> Any:
        ...

    def get_typed(self, config_name: str, schema: Type[T]) -> T:
        ...

    def reload(self, config_name: str) -> dict:
        ...

    @property
    def config_dir(self) -> Path:
        ...
