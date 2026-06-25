from typing import Protocol, Optional


class SecretPort(Protocol):
    def get_secret(self, key: str) -> Optional[str]:
        ...

    def get_api_key(self, provider: str) -> Optional[str]:
        ...

    def require_secret(self, key: str) -> str:
        ...

    def mask_in_logs(self, value: str) -> str:
        ...
