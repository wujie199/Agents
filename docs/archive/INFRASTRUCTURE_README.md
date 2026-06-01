# infrastructure/ 实现层

本目录实现了 `ports/` 中定义的所有 Protocol，是具体的业务逻辑实现。

## 目录结构

```
infrastructure/
├── config/
│   └── adapter.py          # ConfigPortAdapter - YAML 配置加载
├── secret/
│   └── adapter.py          # SecretPortAdapter - 密钥管理
├── privacy/
│   └── adapter.py          # PrivacyPortAdapter - 脱敏实现
├── identity/
│   └── adapter.py          # IdentityPortAdapter - 身份与 ACL
├── policy/
│   └── adapter.py          # PolicyPortAdapter - 并发与批处理策略
└── observability/
    └── adapter.py          # ObservabilityPortAdapter - 追踪与指标

storage/adapters/
├── redis/
│   └── cache_adapter.py    # RedisCacheAdapter / MemoryCacheAdapter
└── chroma/
    └── vector_adapter.py   # ChromaVectorAdapter
```

## 使用示例

### 1. 配置加载

```python
from infrastructure.config.adapter import ConfigPortAdapter

config = ConfigPortAdapter()
llm_config = config.load("llm")
model_name = config.get("llm.chat_model_name")  # → "kimi-k2.6"
```

### 2. 脱敏处理

```python
from infrastructure.privacy.adapter import PrivacyPortAdapter

privacy = PrivacyPortAdapter()

# 日志脱敏
masked = privacy.mask_text("手机号13812345678")  # → "手机号138****5678"

# 审计哈希
args_hash = privacy.hash_for_audit("敏感参数")  # → 16位哈希

# 敏感度分类
level = privacy.classify_sensitivity("身份证370123199001011234")  # → "secret"
```

### 3. 策略控制

```python
from infrastructure.policy.adapter import PolicyPortAdapter

policy = PolicyPortAdapter(config_path="config/concurrency.yml")

# 并发上限
max_sends = policy.get_max_parallel_sends("tenant1")  # → 5

# 批大小建议
batch_size = policy.suggest_batch_size("tenant1", item_count=100)  # → 50
```

### 4. 缓存操作

```python
from storage.adapters.redis.cache_adapter import MemoryCacheAdapter

cache = MemoryCacheAdapter()

# 设置缓存
cache.set("tenant1:qc:hash123", {"result": "data"}, ttl_seconds=900)

# 构建标准键
key = cache.build_key("tenant1", "qc", "query_hash")
# → "tenant1:agents:qc:query_hash"
```

### 5. 向量操作

```python
from storage.adapters.chroma.vector_adapter import ChromaVectorAdapter
from core.ports.storage.vector import VectorRecord

vector = ChromaVectorAdapter(persist_directory="./chroma_db")

# 写入向量
records = [VectorRecord(
    id="doc1_chunk1",
    vector=[0.1, 0.2, ...],
    metadata={"tenant_id": "t1", "doc_id": "doc1"},
    content="文档内容"
)]
vector.upsert("knowledge_base", records)

# 相似检索
results = vector.similarity_search("knowledge_base", query_vector, top_k=10)
```

### 6. 完整集成

```python
from domain import RequestContext
from composition import build_run_context
from infrastructure.privacy.adapter import PrivacyPortAdapter
from infrastructure.policy.adapter import PolicyPortAdapter

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
    policy=PolicyPortAdapter(config_path="config/concurrency.yml")
)

# 使用
masked = ctx.privacy.mask_text("敏感内容")
batch_size = ctx.policy.get_batch_size("tenant1")
```

## 配置文件

| 文件 | 说明 |
|------|------|
| `config/concurrency.yml` | 并发与批处理策略 |
| `config/privacy.yml` | 脱敏规则配置 |

## 测试

```bash
pytest tests/test_infrastructure.py -v
# 23 tests passed
```
