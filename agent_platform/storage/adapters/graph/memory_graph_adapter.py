from typing import Optional, List, Any
from core.ports.storage.graph import GraphPort, GraphNode, GraphEdge, GraphPath


class MemoryGraphAdapter:
    """
    内存图适配器（开发/测试用）
    
    注意：生产环境请使用 Neo4jGraphAdapter
    """
    def __init__(self):
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[tuple[str, str, str], GraphEdge] = {}
        self._node_neighbors: dict[str, List[GraphEdge]] = {}
    
    def upsert_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict,
    ) -> GraphNode:
        existing = self._nodes.get(node_id)
        if existing:
            merged = {**existing.properties, **properties}
            self._nodes[node_id] = GraphNode(
                id=node_id, type=node_type or existing.type, properties=merged
            )
            return self._nodes[node_id]
        node = GraphNode(id=node_id, type=node_type, properties=properties)
        self._nodes[node_id] = node
        self._node_neighbors[node_id] = []
        return node

    def create_node(self, node_type: str, properties: dict) -> GraphNode:
        import uuid
        node_id = str(properties.get("doc_id") or properties.get("id") or uuid.uuid4())[:32]
        if node_id in self._nodes:
            return self.upsert_node(node_id, node_type, properties)
        return self.upsert_node(node_id, node_type, properties)
    
    def create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[dict] = None
    ) -> GraphEdge:
        if source_id not in self._nodes:
            raise ValueError(f"Source node not found: {source_id}")
        if target_id not in self._nodes:
            raise ValueError(f"Target node not found: {target_id}")
        
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            type=edge_type,
            properties=properties or {}
        )
        
        key = (source_id, target_id, edge_type)
        self._edges[key] = edge
        
        self._node_neighbors[source_id].append(edge)
        self._node_neighbors[target_id].append(edge)
        
        return edge
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)
    
    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        direction: str = "both"
    ) -> List[GraphNode]:
        if node_id not in self._nodes:
            return []
        
        neighbors = []
        for edge in self._node_neighbors.get(node_id, []):
            if edge_types and edge.type not in edge_types:
                continue
            
            if direction == "out" and edge.source_id != node_id:
                continue
            if direction == "in" and edge.target_id != node_id:
                continue
            
            neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
            neighbor = self._nodes.get(neighbor_id)
            if neighbor and neighbor not in neighbors:
                neighbors.append(neighbor)
        
        return neighbors
    
    def k_hop_subgraph(
        self,
        node_ids: List[str],
        k: int,
        edge_types: Optional[List[str]] = None
    ) -> List[GraphPath]:
        visited_nodes = set(node_ids)
        current_nodes = set(node_ids)
        
        for _ in range(k):
            next_nodes = set()
            for node_id in current_nodes:
                neighbors = self.get_neighbors(node_id, edge_types=edge_types)
                for neighbor in neighbors:
                    if neighbor.id not in visited_nodes:
                        next_nodes.add(neighbor.id)
                        visited_nodes.add(neighbor.id)
            current_nodes = next_nodes
            if not current_nodes:
                break
        
        nodes = [self._nodes[nid] for nid in visited_nodes if nid in self._nodes]
        edges = [e for e in self._edges.values()
                 if e.source_id in visited_nodes and e.target_id in visited_nodes]
        
        if nodes:
            text_rep = "\n".join([f"[{n.type}] {n.id}" for n in nodes])
            text_rep += "\n" + "\n".join([f"({e.source_id}) --[{e.type}]--> ({e.target_id})" for e in edges])
            return [GraphPath(nodes=nodes, edges=edges, text_representation=text_rep)]
        
        return []
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[GraphPath]:
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        
        from collections import deque
        
        visited = {source_id}
        queue = deque([(source_id, [source_id], [])])
        
        while queue:
            current_id, path_nodes, path_edges = queue.popleft()
            
            if len(path_nodes) > max_depth + 1:
                continue
            
            if current_id == target_id:
                nodes = [self._nodes[nid] for nid in path_nodes]
                return GraphPath(nodes=nodes, edges=path_edges, text_representation="")
            
            for edge in self._node_neighbors.get(current_id, []):
                neighbor_id = edge.target_id if edge.source_id == current_id else edge.source_id
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path_nodes + [neighbor_id], path_edges + [edge]))
        
        return None
    
    def delete_node(self, node_id: str) -> int:
        if node_id not in self._nodes:
            return 0
        
        edges_to_remove = [
            key for key, edge in self._edges.items()
            if edge.source_id == node_id or edge.target_id == node_id
        ]
        
        for key in edges_to_remove:
            del self._edges[key]
        
        del self._nodes[node_id]
        del self._node_neighbors[node_id]
        
        return 1
    
    def delete_edge(self, source_id: str, target_id: str, edge_type: str) -> int:
        key = (source_id, target_id, edge_type)
        if key not in self._edges:
            return 0
        
        edge = self._edges[key]
        
        for nid in [source_id, target_id]:
            if nid in self._node_neighbors:
                self._node_neighbors[nid] = [
                    e for e in self._node_neighbors[nid] if e != edge
                ]
        
        del self._edges[key]
        return 1
    
    def query(self, query_str: str, params: Optional[dict] = None) -> List[Any]:
        return []
    
    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._node_neighbors.clear()
    
    def health(self) -> dict:
        return {
            "status": "healthy",
            "type": "memory_graph",
            "nodes": len(self._nodes),
            "edges": len(self._edges)
        }
