from typing import Optional, List, Any
from dataclasses import dataclass, field
import logging
import time

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
    from neo4j.exceptions import ServiceUnavailable, AuthError
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

from core.ports.storage.graph import GraphPort, GraphNode, GraphEdge, GraphPath


class Neo4jGraphAdapter:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
        connection_timeout: float = 30.0,
        max_transaction_retry_time: float = 30.0
    ):
        if not HAS_NEO4J:
            raise ImportError("neo4j driver not installed. Run: pip install neo4j")
        
        self._uri = uri
        self._database = database
        self._logger = logging.getLogger(__name__)
        
        self._driver: Optional[AsyncDriver] = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=max_connection_pool_size,
            connection_timeout=connection_timeout,
            max_transaction_retry_time=max_transaction_retry_time
        )
        
        self._health_status = "unknown"
    
    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._logger.info("Neo4j driver closed")
    
    @property
    def driver(self) -> AsyncDriver:
        if not self._driver:
            raise RuntimeError("Driver not initialized")
        return self._driver
    
    async def create_node(
        self,
        node_type: str,
        properties: dict
    ) -> GraphNode:
        import uuid
        node_id = properties.get("id", str(uuid.uuid4())[:8])
        
        query = f"""
        MERGE (n:{node_type} {{id: $node_id}})
        SET n += $properties
        RETURN n.id as id, labels(n) as labels, properties(n) as props
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                node_id=node_id,
                properties=properties
            )
            record = await result.single()
            
            if record:
                return GraphNode(
                    id=record["id"],
                    type=record["labels"][0],
                    properties=record["props"]
                )
            
            return GraphNode(id=node_id, type=node_type, properties=properties)
    
    async def create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[dict] = None
    ) -> GraphEdge:
        query = f"""
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        MERGE (source)-[r:{edge_type}]->(target)
        SET r += $properties
        RETURN type(r) as type, properties(r) as props
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                properties=properties or {}
            )
            record = await result.single()
            
            return GraphEdge(
                source_id=source_id,
                target_id=target_id,
                type=edge_type,
                properties=record["props"] if record else (properties or {})
            )
    
    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        query = """
        MATCH (n {id: $node_id})
        RETURN n.id as id, labels(n) as labels, properties(n) as props
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            
            if record:
                return GraphNode(
                    id=record["id"],
                    type=record["labels"][0] if record["labels"] else "Node",
                    properties=record["props"]
                )
            return None
    
    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        direction: str = "both"
    ) -> List[GraphNode]:
        if edge_types:
            edge_pattern = "|".join(edge_types)
        else:
            edge_pattern = "r"
        
        if direction == "out":
            query = f"""
            MATCH (n {{id: $node_id}})-[:{edge_pattern}]->(neighbor)
            RETURN DISTINCT neighbor.id as id, labels(neighbor) as labels, 
                   properties(neighbor) as props
            """
        elif direction == "in":
            query = f"""
            MATCH (n {{id: $node_id}})<-[:{edge_pattern}]-(neighbor)
            RETURN DISTINCT neighbor.id as id, labels(neighbor) as labels, 
                   properties(neighbor) as props
            """
        else:
            query = f"""
            MATCH (n {{id: $node_id}})-[:{edge_pattern}]-(neighbor)
            RETURN DISTINCT neighbor.id as id, labels(neighbor) as labels, 
                   properties(neighbor) as props
            """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, node_id=node_id)
            records = await result.fetch()
            
            return [
                GraphNode(
                    id=r["id"],
                    type=r["labels"][0] if r["labels"] else "Node",
                    properties=r["props"]
                )
                for r in records
            ]
    
    async def k_hop_subgraph(
        self,
        node_ids: List[str],
        k: int,
        edge_types: Optional[List[str]] = None
    ) -> List[GraphPath]:
        if edge_types:
            edge_pattern = "|".join(edge_types)
        else:
            edge_pattern = ""
        
        query = f"""
        MATCH path = (n)-[*1..{k} {edge_pattern}]-(m)
        WHERE n.id IN $node_ids
        WITH collect(DISTINCT n) + collect(DISTINCT m) as all_nodes,
             collect(DISTINCT relationships(path)) as all_rels
        UNWIND all_nodes as node
        WITH collect(DISTINCT {{id: node.id, type: labels(node)[0], 
                props: properties(node)}}) as nodes, all_rels
        UNWIND all_rels as rels
        UNWIND rels as rel
        WITH nodes, collect(DISTINCT {{source: startNode(rel).id, 
                target: endNode(rel).id, type: type(rel), 
                props: properties(rel)}}) as edges
        RETURN nodes, edges
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, node_ids=node_ids)
            record = await result.single()
            
            if not record:
                return []
            
            nodes = [
                GraphNode(id=n["id"], type=n["type"], properties=n["props"])
                for n in record["nodes"]
            ]
            
            edges = [
                GraphEdge(
                    source_id=e["source"],
                    target_id=e["target"],
                    type=e["type"],
                    properties=e["props"]
                )
                for e in record["edges"]
            ]
            
            text_rep = self._path_to_text(nodes, edges)
            
            return [GraphPath(nodes=nodes, edges=edges, text_representation=text_rep)]
    
    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[GraphPath]:
        query = f"""
        MATCH path = shortestPath(
            (source {{id: $source_id}})-[*1..{max_depth}]-(target {{id: $target_id}})
        )
        RETURN nodes(path) as path_nodes, relationships(path) as path_rels
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                source_id=source_id,
                target_id=target_id
            )
            record = await result.single()
            
            if not record:
                return None
            
            nodes = [
                GraphNode(
                    id=n.id,
                    type=list(n.labels)[0] if n.labels else "Node",
                    properties=dict(n)
                )
                for n in record["path_nodes"]
            ]
            
            edges = [
                GraphEdge(
                    source_id=r.start_node.id,
                    target_id=r.end_node.id,
                    type=r.type,
                    properties=dict(r)
                )
                for r in record["path_rels"]
            ]
            
            return GraphPath(
                nodes=nodes,
                edges=edges,
                text_representation=self._path_to_text(nodes, edges)
            )
    
    def _path_to_text(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> str:
        lines = []
        for node in nodes:
            lines.append(f"[{node.type}] {node.id}")
        for edge in edges:
            lines.append(f"({edge.source_id}) --[{edge.type}]--> ({edge.target_id})")
        return "\n".join(lines)
    
    async def delete_node(self, node_id: str) -> int:
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            return record["deleted"] if record else 0
    
    async def delete_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str
    ) -> int:
        query = f"""
        MATCH (source {{id: $source_id}})-[r:{edge_type}]->(target {{id: $target_id}})
        DELETE r
        RETURN count(r) as deleted
        """
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                source_id=source_id,
                target_id=target_id
            )
            record = await result.single()
            return record["deleted"] if record else 0
    
    async def query(
        self,
        query_str: str,
        params: Optional[dict] = None
    ) -> List[Any]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query_str, params or {})
            records = await result.fetch()
            return [dict(r) for r in records]
    
    async def health(self) -> dict:
        try:
            start = time.time()
            async with self._driver.session(database=self._database) as session:
                await session.run("RETURN 1")
            
            latency = (time.time() - start) * 1000
            
            self._health_status = "healthy"
            return {
                "status": "healthy",
                "type": "neo4j",
                "uri": self._uri,
                "database": self._database,
                "latency_ms": round(latency, 2)
            }
            
        except ServiceUnavailable as e:
            self._health_status = "unhealthy"
            return {
                "status": "unhealthy",
                "error": "Service unavailable",
                "details": str(e)
            }
        except AuthError as e:
            self._health_status = "unhealthy"
            return {
                "status": "unhealthy",
                "error": "Authentication failed",
                "details": str(e)
            }
        except Exception as e:
            self._health_status = "unhealthy"
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def create_indexes(self) -> None:
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Person) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.id)",
        ]
        
        async with self._driver.session(database=self._database) as session:
            for query in queries:
                try:
                    await session.run(query)
                except Exception as e:
                    self._logger.warning(f"Index creation failed: {e}")
        
        self._logger.info("Indexes created")
