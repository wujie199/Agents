from typing import Optional, TypeVar, Callable
from core.domain.context import RequestContext, ACL
from core.composition.run_context import RunContext


T = TypeVar("T")


class FakeConfigPort:
    def __init__(self, configs: Optional[dict] = None):
        self._configs = configs or {}
        self._config_dir = None

    def load(self, config_name: str) -> dict:
        return self._configs.get(config_name, {})

    def get(self, key: str, default: any = None) -> any:
        keys = key.split(".")
        value = self._configs
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def get_typed(self, config_name: str, schema: type) -> any:
        data = self.load(config_name)
        if hasattr(schema, "from_dict"):
            return schema.from_dict(data)
        return schema(**data)

    def reload(self, config_name: str) -> dict:
        return self.load(config_name)

    @property
    def config_dir(self):
        from pathlib import Path
        return self._config_dir or Path("config")


class FakeSecretPort:
    def __init__(self, secrets: Optional[dict] = None):
        self._secrets = secrets or {}

    def get_secret(self, key: str) -> Optional[str]:
        return self._secrets.get(key)

    def get_api_key(self, provider: str) -> Optional[str]:
        return self._secrets.get(f"{provider}_api_key")

    def require_secret(self, key: str) -> str:
        value = self.get_secret(key)
        if value is None:
            raise ValueError(f"Secret not found: {key}")
        return value

    def mask_in_logs(self, value: str) -> str:
        if len(value) <= 4:
            return "****"
        return value[:2] + "****" + value[-2:]


class FakePrivacyPort:
    def mask_text(self, text: str, policy: Optional[str] = None) -> str:
        import re
        text = re.sub(r'1[3-9]\d{9}', lambda m: m.group()[:3] + '****' + m.group()[-4:], text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+', lambda m: m.group()[0] + '***@' + m.group().split('@')[1], text)
        return text

    def redact_for_storage(self, record: dict) -> dict:
        return {k: self.mask_text(str(v)) if isinstance(v, str) else v for k, v in record.items()}

    def redact_for_llm(self, messages: list, policy: Optional[str] = None) -> list:
        return [{"role": m.get("role"), "content": self.mask_text(m.get("content", ""))} for m in messages]

    def hash_for_audit(self, value: str) -> str:
        import hashlib
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def classify_sensitivity(self, text: str) -> str:
        import re
        if re.search(r'1[3-9]\d{9}', text) or re.search(r'[\w\.-]+@[\w\.-]+', text):
            return "pii"
        return "public"


class FakeIdentityPort:
    def resolve_acl(self, tenant_id: str, user_id: str, channel: str) -> ACL:
        return ACL(
            doc_ids=frozenset(),
            tool_names=frozenset(["read_json_all_title", "save_result_2_json"]),
            mcp_servers=frozenset()
        )

    def validate_tenant(self, tenant_id: str) -> bool:
        return bool(tenant_id)

    def validate_user(self, tenant_id: str, user_id: str) -> bool:
        return bool(tenant_id and user_id)


class FakePolicyPort:
    def __init__(
        self,
        max_parallel_sends: int = 5,
        max_batch_workers: int = 3,
        default_batch_size: int = 10
    ):
        self._max_parallel_sends = max_parallel_sends
        self._max_batch_workers = max_batch_workers
        self._default_batch_size = default_batch_size

    def check_rate_limit(self, tenant_id: str, user_id: str, action: str):
        from core.ports.policy import PolicyResult
        return PolicyResult(allowed=True)

    def check_token_budget(self, tenant_id: str, user_id: str, requested_tokens: int):
        from core.ports.policy import PolicyResult
        return PolicyResult(allowed=True)

    def get_max_parallel_sends(self, tenant_id: str) -> int:
        return self._max_parallel_sends

    def get_max_intra_batch_workers(self, tenant_id: str) -> int:
        return self._max_batch_workers

    def get_batch_size(self, tenant_id: str, context: Optional[dict] = None) -> int:
        return self._default_batch_size

    def suggest_batch_size(self, tenant_id: str, item_count: int, context: Optional[dict] = None) -> int:
        return min(item_count, self._default_batch_size)


class FakeObservabilityPort:
    def __init__(self):
        self._spans = {}
        self._events = []

    def start_span(self, trace_id: str, name: str, layer, parent_span_id: Optional[str] = None, attributes: Optional[dict] = None):
        from core.ports.observability import Span
        from datetime import datetime
        import uuid
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4())[:8],
            parent_span_id=parent_span_id,
            name=name,
            layer=layer,
            start_time=datetime.now(),
            attributes=attributes or {}
        )
        self._spans[span.span_id] = span
        return span

    def end_span(self, span, attributes: Optional[dict] = None):
        from datetime import datetime
        span.end_time = datetime.now()
        if attributes:
            span.attributes.update(attributes)

    def log_event(self, trace_id: str, event: str, layer, attributes: Optional[dict] = None):
        self._events.append({
            "trace_id": trace_id,
            "event": event,
            "layer": layer,
            "attributes": attributes or {}
        })

    def record_metric(self, name: str, value: float, tags: Optional[dict] = None):
        pass


class FakeModelPort:
    def __init__(self, models: Optional[dict] = None):
        self._models = models or {}
        self._cache = {}

    def get_model(self, role: str):
        if role in self._cache:
            return self._cache[role]
        model = self._models.get(role)
        if model is None:
            raise ValueError(f"Model not found for role: {role}")
        self._cache[role] = model
        return model

    def get_model_info(self, role: str):
        from core.ports.model import ModelInfo
        return ModelInfo(
            role=role,
            profile="fake",
            provider="fake",
            degraded=False,
            circuit_open=False,
            fallback_index=0
        )

    def invalidate_cache(self, role: Optional[str] = None):
        if role:
            self._cache.pop(role, None)
        else:
            self._cache.clear()

    def get_embedding(self, role: str = "embedding"):
        return self.get_model(role)

    def get_reranker(self, role: str = "rerank"):
        return self.get_model(role)


def build_run_context(
    request: RequestContext,
    rag=None,
    memory=None,
    tools=None,
    skills=None,
    mcp=None,
    models=None,
    policy=None,
    privacy=None,
    observability=None,
    identity=None,
    **extra
) -> RunContext:
    return RunContext(
        request=request,
        rag=rag,
        memory=memory,
        tools=tools,
        skills=skills,
        mcp=mcp,
        models=models or FakeModelPort(),
        policy=policy or FakePolicyPort(),
        privacy=privacy or FakePrivacyPort(),
        observability=observability or FakeObservabilityPort(),
        identity=identity or FakeIdentityPort(),
        extra=extra
    )


def build_test_context(
    tenant_id: str = "test_tenant",
    user_id: str = "test_user",
    session_id: str = "test_session",
    trace_id: str = "test_trace",
    channel: str = "test",
    **kwargs
) -> RunContext:
    request = RequestContext(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        trace_id=trace_id,
        channel=channel
    )
    return build_run_context(request=request, **kwargs)
