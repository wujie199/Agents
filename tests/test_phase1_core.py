import pytest
from core.domain.context import RequestContext, ACL
from core.domain.evidence import EvidenceBundle, Evidence, SourceType, DegradedReason
from core.domain.task import AgentTask
from core.composition.factory import (
    build_test_context,
    build_run_context,
    FakePrivacyPort,
    FakePolicyPort,
)


class TestRequestContext:
    def test_create_context(self):
        ctx = RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr1",
            channel="web"
        )
        assert ctx.tenant_id == "t1"
        assert ctx.user_id == "u1"

    def test_required_fields(self):
        with pytest.raises(ValueError):
            RequestContext(
                tenant_id="",
                user_id="u1",
                session_id="s1",
                trace_id="tr1",
                channel="web"
            )

    def test_acl_permissions(self):
        acl = ACL(
            doc_ids=frozenset(["doc1", "doc2"]),
            tool_names=frozenset(["tool1"]),
            mcp_servers=frozenset(["mcp1"])
        )
        assert acl.can_access_doc("doc1")
        assert not acl.can_access_doc("doc3")
        assert acl.can_use_tool("tool1")
        assert acl.can_use_mcp("mcp1")


class TestEvidenceBundle:
    def test_empty_bundle(self):
        bundle = EvidenceBundle(evidences=[])
        assert bundle.empty is True

    def test_empty_bundle_factory(self):
        bundle = EvidenceBundle.empty_bundle(
            reason=DegradedReason.ALL_BACKENDS_FAILED,
            error_code="RAG_001"
        )
        assert bundle.empty is True
        assert bundle.degraded_reason == DegradedReason.ALL_BACKENDS_FAILED
        assert bundle.error_code == "RAG_001"

    def test_with_evidences(self):
        evidence = Evidence(
            id="e1",
            content="test content",
            source_type=SourceType.VECTOR,
            score=0.9
        )
        bundle = EvidenceBundle(evidences=[evidence])
        assert bundle.empty is False
        assert bundle.total_content_length() == 12


class TestAgentTask:
    def test_create_task(self):
        task = AgentTask(task_id="task1", intent="write_report")
        assert task.task_id == "task1"
        assert task.intent == "write_report"

    def test_thread_id_generation(self):
        task = AgentTask(task_id="task1", intent="test")
        thread_id = task.to_thread_id("session1")
        assert thread_id == "session1:task1"


class TestRunContext:
    def test_build_test_context(self):
        ctx = build_test_context()
        assert ctx.request.tenant_id == "test_tenant"
        assert ctx.policy is not None
        assert ctx.privacy is not None

    def test_get_model(self):
        ctx = build_test_context()
        with pytest.raises(ValueError):
            ctx.get_model("main_llm")

    def test_require_rag_raises(self):
        ctx = build_test_context()
        with pytest.raises(RuntimeError):
            ctx.require_rag()


class TestFakePorts:
    def test_privacy_mask(self):
        port = FakePrivacyPort()
        masked = port.mask_text("æçææºæ¯13812345678")
        assert "****" in masked

    def test_privacy_hash(self):
        port = FakePrivacyPort()
        hash1 = port.hash_for_audit("test")
        hash2 = port.hash_for_audit("test")
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_policy_defaults(self):
        port = FakePolicyPort()
        assert port.get_max_parallel_sends("t1") == 5
        assert port.get_batch_size("t1") == 10

    def test_policy_suggest_batch_size(self):
        port = FakePolicyPort(default_batch_size=10)
        assert port.suggest_batch_size("t1", 5) == 5
        assert port.suggest_batch_size("t1", 20) == 10
