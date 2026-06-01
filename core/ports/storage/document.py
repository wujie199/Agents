from typing import Protocol, List, Optional, Any, Dict
from datetime import datetime


class DocumentPort(Protocol):
    """文档数据库 Port（MongoDB）
    
    适用场景：
    - 对话历史存储（灵活 schema）
    - 配置存储（JSON 文档）
    - 日志存储（时序数据）
    - 用户画像（嵌套文档）
    """
    
    async def insert_one(
        self,
        collection: str,
        document: Dict[str, Any]
    ) -> str:
        ...
    
    async def insert_many(
        self,
        collection: str,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        ...
    
    async def find_one(
        self,
        collection: str,
        filter: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        ...
    
    async def find_many(
        self,
        collection: str,
        filter: Dict[str, Any],
        sort: Optional[List[tuple]] = None,
        skip: int = 0,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        ...
    
    async def update_one(
        self,
        collection: str,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False
    ) -> bool:
        ...
    
    async def update_many(
        self,
        collection: str,
        filter: Dict[str, Any],
        update: Dict[str, Any]
    ) -> int:
        ...
    
    async def delete_one(
        self,
        collection: str,
        filter: Dict[str, Any]
    ) -> bool:
        ...
    
    async def delete_many(
        self,
        collection: str,
        filter: Dict[str, Any]
    ) -> int:
        ...
    
    async def count(
        self,
        collection: str,
        filter: Dict[str, Any]
    ) -> int:
        ...
    
    async def aggregate(
        self,
        collection: str,
        pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        ...
    
    async def create_index(
        self,
        collection: str,
        keys: List[tuple],
        unique: bool = False
    ) -> str:
        ...
