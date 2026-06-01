import pytest
import asyncio
from core.domain.context import RequestContext
from agent_platform.storage.adapters.sqlite.relational_adapter import AsyncSQLiteRelationalAdapter
from agent_platform.storage.adapters.graph.memory_graph_adapter import MemoryGraphAdapter
from agent_platform.infrastructure.mcp.adapter import EnterpriseMCPAdapter


class TestAsyncSQLiteRelationalAdapter:
    @pytest.mark.asyncio
    async def test_insert_and_select(self):
        adapter = AsyncSQLiteRelationalAdapter(db_path="data/test_archive.db")
        
        try:
            session_id = await adapter.insert("sessions", {
                "session_id": "test_session_async",
                "user_id": "user1",
                "tenant_id": "tenant1",
                "channel": "web",
                "started_at": "2024-01-01T00:00:00",
                "status": "active"
            })
            
            result = await adapter.select_one(
                "sessions",
                ["session_id", "user_id", "tenant_id"],
                {"session_id": "test_session_async"}
            )
            
            assert result is not None
            assert result["user_id"] == "user1"
        finally:
            await adapter.close()
    
    @pytest.mark.asyncio
    async def test_health(self):
        adapter = AsyncSQLiteRelationalAdapter(db_path="data/test_archive.db")
        
        try:
            health = await adapter.health()
            assert health["status"] == "healthy"
        finally:
            await adapter.close()


class TestMemoryGraphAdapter:
    def test_create_nodes_and_edges(self):
        adapter = MemoryGraphAdapter()
        
        node1 = adapter.create_node("person", {"name": "Alice"})
        node2 = adapter.create_node("person", {"name": "Bob"})
        
        assert node1.id is not None
        assert node1.type == "person"
        
        edge = adapter.create_edge(node1.id, node2.id, "knows", {"since": "2020"})
        assert edge.type == "knows"
    
    def test_get_neighbors(self):
        adapter = MemoryGraphAdapter()
        
        node1 = adapter.create_node("person", {"name": "Alice"})
        node2 = adapter.create_node("person", {"name": "Bob"})
        node3 = adapter.create_node("person", {"name": "Charlie"})
        
        adapter.create_edge(node1.id, node2.id, "knows")
        adapter.create_edge(node1.id, node3.id, "knows")
        
        neighbors = adapter.get_neighbors(node1.id)
        assert len(neighbors) == 2
    
    def test_find_path(self):
        adapter = MemoryGraphAdapter()
        
        node1 = adapter.create_node("person", {"name": "Alice"})
        node2 = adapter.create_node("person", {"name": "Bob"})
        node3 = adapter.create_node("person", {"name": "Charlie"})
        
        adapter.create_edge(node1.id, node2.id, "knows")
        adapter.create_edge(node2.id, node3.id, "knows")
        
        path = adapter.find_path(node1.id, node3.id, max_depth=3)
        assert path is not None
        assert len(path.nodes) == 3
    
    def test_k_hop_subgraph(self):
        adapter = MemoryGraphAdapter()
        
        node1 = adapter.create_node("person", {"name": "Alice"})
        node2 = adapter.create_node("person", {"name": "Bob"})
        node3 = adapter.create_node("person", {"name": "Charlie"})
        
        adapter.create_edge(node1.id, node2.id, "knows")
        adapter.create_edge(node2.id, node3.id, "knows")
        
        paths = adapter.k_hop_subgraph([node1.id], k=2)
        assert len(paths) == 1
        assert len(paths[0].nodes) == 3


class TestEnterpriseMCPAdapter:
    def test_list_servers(self):
        adapter = EnterpriseMCPAdapter()
        servers = adapter.list_servers()
        assert isinstance(servers, list)
    
    def test_get_server_info_not_found(self):
        adapter = EnterpriseMCPAdapter()
        info = adapter.get_server_info("nonexistent")
        assert info is None


class TestProductionContext:
    @pytest.mark.asyncio
    async def test_build_production_context(self):
        from core.composition.production_factory import build_production_context
        
        request = RequestContext(
            tenant_id="tenant1",
            user_id="user1",
            session_id="session1",
            trace_id="trace1",
            channel="web"
        )
        
        ctx = build_production_context(
            request,
            use_memory_cache=True,
            use_memory_graph=True
        )
        
        assert ctx.privacy is not None
        assert ctx.policy is not None
        assert ctx.observability is not None
        assert ctx.identity is not None
        assert ctx.skills is not None
        assert ctx.mcp is not None
        
        assert "cache" in ctx.extra
        assert "relational" in ctx.extra
        assert "graph" in ctx.extra
        assert "object_store" in ctx.extra
        
        if hasattr(ctx.extra["relational"], "close"):
            await ctx.extra["relational"].close()
    
    def test_use_context_ports(self):
        from core.composition.production_factory import build_production_context
        
        request = RequestContext(
            tenant_id="tenant1",
            user_id="user1",
            session_id="session1",
            trace_id="trace1",
            channel="web"
        )
        
        ctx = build_production_context(
            request,
            use_memory_cache=True,
            use_memory_graph=True
        )
        
        masked = ctx.privacy.mask_text("ææº13812345678")
        assert "****" in masked
        
        batch_size = ctx.policy.get_batch_size("tenant1")
        assert batch_size > 0
    
    def test_build_development_context(self):
        from core.composition.production_factory import build_development_context
        
        request = RequestContext(
            tenant_id="tenant1",
            user_id="user1",
            session_id="session1",
            trace_id="trace1",
            channel="web"
        )
        
        ctx = build_development_context(request)
        
        assert ctx.privacy is not None
        assert ctx.policy is not None
        assert "cache" in ctx.extra
        assert "graph" in ctx.extra
