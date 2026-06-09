import pytest
import asyncio
from core.domain.context import RequestContext
from agent_platform.model.registry import ModelRegistry
from document.rag.bridges.rag_port_adapter import RAGPortAdapter
from agent_platform.tools.adapters.tool_port_adapter import ToolPortAdapter
from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter


class TestModelRegistry:
    def test_load_config(self):
        registry = ModelRegistry(config_path="config/models.yml")
        assert registry is not None

    def test_main_llm_cloud_fallback_when_no_local(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_ROOT", str(tmp_path / "none"))
        monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:59999/v1")
        registry = ModelRegistry(config_path="config/models.yml")
        role = registry._roles.get("main_llm")
        assert role is not None
        assert role.profile == "dashscope_main"
        assert registry._profiles["dashscope_main"].model_name == "qwen3.6-plus"

    def test_main_llm_local_hf_when_weights_present(self, tmp_path, monkeypatch):
        root = tmp_path / "llm"
        model = root / "Qwen3-0.6B"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"x")
        monkeypatch.setenv("LOCAL_LLM_ROOT", str(root))
        registry = ModelRegistry(config_path="config/models.yml")
        role = registry._roles.get("main_llm")
        assert role.profile == "local_hf_chat"
        assert "dashscope_main" in (role.fallback_chain or [])

    def test_get_model_info(self):
        registry = ModelRegistry(config_path="config/models.yml")
        
        try:
            info = registry.get_model_info("main_llm")
            assert info.role == "main_llm"
        except ValueError:
            pytest.skip("main_llm role not configured")
    
    def test_health(self):
        registry = ModelRegistry(config_path="config/models.yml")
        health = registry.health()
        assert "roles" in health


class TestRAGPortAdapter:
    def test_init(self):
        vector_port = ChromaVectorAdapter(persist_directory="data/test_chroma")
        
        rag = RAGPortAdapter(
            vector_port=vector_port,
            enable_cache=False
        )
        
        assert rag is not None
    
    @pytest.mark.asyncio
    async def test_health(self):
        vector_port = ChromaVectorAdapter(persist_directory="data/test_chroma")
        
        rag = RAGPortAdapter(
            vector_port=vector_port,
            enable_cache=False
        )
        
        health = await rag.health()
        assert "status" in health


class TestToolPortAdapter:
    def test_init(self):
        tools = ToolPortAdapter(config_path="config/tools.yml")
        assert tools is not None
    
    def test_list_tools(self):
        tools = ToolPortAdapter(config_path="config/tools.yml")
        tool_list = tools.list_tools()
        assert isinstance(tool_list, list)
    
    def test_health(self):
        tools = ToolPortAdapter(config_path="config/tools.yml")
        health = tools.health()
        assert health["status"] == "healthy"


class TestMemoryPortAdapter:
    def test_init(self):
        memory = MemoryPortAdapter(store_dir="data/test_memory")
        assert memory is not None
    
    def test_compose_prompt_snapshot(self):
        memory = MemoryPortAdapter(store_dir="data/test_memory")
        
        request = RequestContext(
            tenant_id="tenant1",
            user_id="user1",
            session_id="session1",
            trace_id="trace1",
            channel="web"
        )
        
        snapshot = memory.compose_prompt_snapshot(request)
        
        assert snapshot.memory_text is not None
        assert snapshot.hash is not None
        assert snapshot.frozen is True
    
    def test_health(self):
        memory = MemoryPortAdapter(store_dir="data/test_memory")
        health = memory.health()
        assert health["status"] == "healthy"


class TestFullContext:
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
            use_memory_graph=True
        )
        
        assert ctx.rag is not None
        assert ctx.memory is not None
        assert ctx.tools is not None
        assert ctx.models is not None
        assert ctx.mcp is not None
        assert ctx.policy is not None
        assert ctx.privacy is not None
        
        if hasattr(ctx.extra.get("relational"), "close"):
            await ctx.extra["relational"].close()
    
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
        
        assert ctx.rag is not None
        assert ctx.memory is not None
        assert ctx.tools is not None
        assert ctx.models is not None
