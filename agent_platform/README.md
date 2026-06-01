# agent_platform — 平台层（统一入口）

> 包名使用 `agent_platform`，避免与 Python 标准库 `platform` 冲突。

| 子包 | 职责 |
|------|------|
| `storage/` | 向量、缓存、SQL、图、对象存储 |
| `infrastructure/` | 配置、密钥、隐私、身份、策略、观测、MCP、Skills |
| `model/` | LLM 注册与 Provider |
| `memory/` | 会话记忆适配 |
| `tools/` | 工具调用适配 |

```python
from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter
from agent_platform.infrastructure.config.adapter import ConfigPortAdapter
from agent_platform.model.registry import ModelRegistry
```
