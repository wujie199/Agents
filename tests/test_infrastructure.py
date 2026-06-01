import pytest
from core.domain.context import RequestContext, ACL
from core.composition.factory import build_run_context
from agent_platform.infrastructure.config.adapter import ConfigPortAdapter
from agent_platform.infrastructure.secret.adapter import SecretPortAdapter
from agent_platform.infrastructure.privacy.adapter import PrivacyPortAdapter
from agent_platform.infrastructure.identity.adapter import IdentityPortAdapter
from agent_platform.infrastructure.policy.adapter import PolicyPortAdapter
from agent_platform.infrastructure.observability.adapter import ObservabilityPortAdapter
from agent_platform.storage.adapters.redis.cache_adapter import MemoryCacheAdapter


class TestConfigPortAdapter:
    def test_load_config(self):
        adapter = ConfigPortAdapter()
        config = adapter.load("llm")
        assert "chat_model_name" in config
    
    def test_get_nested_value(self):
        adapter = ConfigPortAdapter()
        model_name = adapter.get("llm.chat_model_name")
        assert model_name == "kimi-k2.6"
    
    def test_get_with_default(self):
        adapter = ConfigPortAdapter()
        value = adapter.get("llm.nonexistent", default="default_value")
        assert value == "default_value"


class TestSecretPortAdapter:
    def test_get_secret_from_dict(self):
        adapter = SecretPortAdapter(secrets={"api_key": "secret123"})
        assert adapter.get_secret("api_key") == "secret123"
    
    def test_mask_in_logs(self):
        adapter = SecretPortAdapter()
        masked = adapter.mask_in_logs("sk-d922c27b5c4b4ed284e30ae0ba70743f")
        assert "****" in masked
        assert masked.startswith("sk")
    
    def test_require_secret_raises(self):
        adapter = SecretPortAdapter()
        with pytest.raises(ValueError):
            adapter.require_secret("nonexistent")


class TestPrivacyPortAdapter:
    def test_mask_phone(self):
        adapter = PrivacyPortAdapter()
        masked = adapter.mask_text("ææºå·13812345678")
        assert "****" in masked
        assert "138" in masked
        assert "5678" in masked
    
    def test_mask_email(self):
        adapter = PrivacyPortAdapter()
        masked = adapter.mask_text("é®ç®±test@example.com")
        assert "***@" in masked
    
    def test_hash_for_audit(self):
        adapter = PrivacyPortAdapter()
        hash1 = adapter.hash_for_audit("sensitive_data")
        hash2 = adapter.hash_for_audit("sensitive_data")
        assert hash1 == hash2
        assert len(hash1) == 16
    
    def test_classify_sensitivity(self):
        adapter = PrivacyPortAdapter()
        assert adapter.classify_sensitivity("æ®éææ¬").value == "public"
        assert adapter.classify_sensitivity("ææº13812345678").value == "pii"
    
    def test_redact_for_storage(self):
        adapter = PrivacyPortAdapter()
        record = {"name": "å¼ ä¸", "phone": "13812345678"}
        redacted = adapter.redact_for_storage(record)
        assert "****" in redacted["phone"]


class TestIdentityPortAdapter:
    def test_validate_tenant(self):
        adapter = IdentityPortAdapter()
        assert adapter.validate_tenant("tenant1") is True
        assert adapter.validate_tenant("") is False
    
    def test_resolve_acl(self):
        adapter = IdentityPortAdapter()
        acl = adapter.resolve_acl("tenant1", "user1", "web")
        assert isinstance(acl, ACL)
        assert "read_json_all_title" in acl.tool_names


class TestPolicyPortAdapter:
    def test_default_config(self):
        adapter = PolicyPortAdapter()
        assert adapter.get_max_parallel_sends("tenant1") == 5
        assert adapter.get_batch_size("tenant1") == 10
    
    def test_suggest_batch_size(self):
        adapter = PolicyPortAdapter()
        assert adapter.suggest_batch_size("tenant1", 5) == 5
        assert adapter.suggest_batch_size("tenant1", 100) == 50  # max_batch_size
    
    def test_load_from_config(self):
        adapter = PolicyPortAdapter(config_path="config/concurrency.yml")
        assert adapter.get_max_parallel_sends("tenant1") == 5


class TestObservabilityPortAdapter:
    def test_start_end_span(self):
        from core.ports.observability import Layer
        adapter = ObservabilityPortAdapter()
        
        span = adapter.start_span(
            trace_id="trace1",
            name="test_span",
            layer=Layer.AGENT
        )
        assert span.trace_id == "trace1"
        assert span.end_time is None
        
        adapter.end_span(span)
        assert span.end_time is not None
    
    def test_record_metric(self):
        adapter = ObservabilityPortAdapter()
        adapter.record_metric("latency", 100.5)
        adapter.record_metric("latency", 200.5)
        
        stats = adapter.get_metric_stats("latency")
        assert stats["count"] == 2
        assert stats["avg"] == 150.5


class TestMemoryCacheAdapter:
    def test_set_get(self):
        adapter = MemoryCacheAdapter()
        adapter.set("key1", {"data": "value1"})
        result = adapter.get("key1")
        assert result == {"data": "value1"}
    
    def test_ttl(self):
        adapter = MemoryCacheAdapter()
        adapter.set("key1", "value1", ttl_seconds=1)
        assert adapter.get("key1") == "value1"
    
    def test_build_key(self):
        adapter = MemoryCacheAdapter()
        key = adapter.build_key("tenant1", "qc", "query_hash")
        assert key == "tenant1:agents:qc:query_hash"
    
    def test_invalidate_pattern(self):
        adapter = MemoryCacheAdapter()
        adapter.set("test:key1", "value1")
        adapter.set("test:key2", "value2")
        adapter.set("other:key3", "value3")
        
        count = adapter.invalidate_pattern("test:*")
        assert count == 2


class TestIntegration:
    def test_build_run_context_with_adapters(self):
        request = RequestContext(
            tenant_id="tenant1",
            user_id="user1",
            session_id="session1",
            trace_id="trace1",
            channel="web"
        )
        
        ctx = build_run_context(
            request=request,
            privacy=PrivacyPortAdapter(),
            policy=PolicyPortAdapter(config_path="config/concurrency.yml"),
            observability=ObservabilityPortAdapter()
        )
        
        masked = ctx.privacy.mask_text("ææº13812345678")
        assert "****" in masked
        
        batch_size = ctx.policy.get_batch_size("tenant1")
        assert batch_size == 10
