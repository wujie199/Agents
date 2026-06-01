from typing import Protocol, Optional, BinaryIO
from dataclasses import dataclass


@dataclass
class ObjectMetadata:
    key: str
    size: int
    content_type: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None


class ObjectStorePort(Protocol):
    def upload(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None
    ) -> str:
        ...
    
    def upload_file(
        self,
        key: str,
        file_path: str
    ) -> str:
        ...
    
    def download(self, key: str) -> Optional[bytes]:
        ...
    
    def download_to_file(
        self,
        key: str,
        file_path: str
    ) -> bool:
        ...
    
    def delete(self, key: str) -> bool:
        ...
    
    def exists(self, key: str) -> bool:
        ...
    
    def get_metadata(self, key: str) -> Optional[ObjectMetadata]:
        ...
    
    def list_objects(
        self,
        prefix: str,
        limit: Optional[int] = None
    ) -> list[ObjectMetadata]:
        ...
    
    def get_signed_url(
        self,
        key: str,
        expires_in: int = 3600
    ) -> Optional[str]:
        ...
