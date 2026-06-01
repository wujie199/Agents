from typing import Protocol, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class GraphNode:
    id: str
    type: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    type: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphPath:
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    text_representation: Optional[str] = None


class GraphPort(Protocol):
    def create_node(
        self,
        node_type: str,
        properties: dict
    ) -> GraphNode:
        ...
    
    def create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[dict] = None
    ) -> GraphEdge:
        ...
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        ...
    
    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        direction: str = "both"
    ) -> List[GraphNode]:
        ...
    
    def k_hop_subgraph(
        self,
        node_ids: List[str],
        k: int,
        edge_types: Optional[List[str]] = None
    ) -> List[GraphPath]:
        ...
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[GraphPath]:
        ...
    
    def delete_node(self, node_id: str) -> int:
        ...
    
    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str
    ) -> int:
        ...
    
    def query(
        self,
        query_str: str,
        params: Optional[dict] = None
    ) -> List[Any]:
        ...
