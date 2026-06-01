# 设计模式实例说明

基于项目代码中的实际应用，本文档说明每种设计模式的简单示例、解决的问题、原理和好处。

---

## 一、策略模式

### 代码示例

从 `knowledge/query/router/fusion.py` 中提取：

```python
from abc import ABC, abstractmethod

class FusionStrategy(ABC):
    """融合策略抽象基类"""
    
    @abstractmethod
    def fuse(self, results, weights=None):
        """定义融合算法接口"""
        ...


class RRFFusion(FusionStrategy):
    """倒数排名融合策略"""
    def __init__(self, k: int = 60):
        self._k = k
    
    def fuse(self, results, weights=None):
        scores = {}
        for list_idx, evidence_list in enumerate(results):
            weight = weights[list_idx] if weights else 1.0
            for rank, evidence in enumerate(evidence_list, start=1):
                doc_id = evidence.id
                rrf_score = weight / (self._k + rank)
                scores[doc_id] = scores.get(doc_id, 0) + rrf_score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class WeightedFusion(FusionStrategy):
    """加权融合策略"""
    def fuse(self, results, weights=None):
        # 对每个结果列表的分数进行加权求和
        ...


class CascadeFusion(FusionStrategy):
    """级联融合策略"""
    def fuse(self, results, weights=None):
        # 按优先级顺序使用各后端结果
        ...
```

### 解决的问题

当系统需要支持多种算法实现，且这些算法可以在运行时动态切换时，如果直接在业务代码中使用 if-else 分支判断，会导致代码耦合度高、扩展性差。策略模式通过定义统一的算法接口，让不同算法实现可以相互替换，避免修改业务代码。

### 原理

策略模式的核心原理是"开闭原则"——对扩展开放，对修改关闭。通过定义抽象策略接口（FusionStrategy），将算法的实现细节封装在具体策略类中（RRFFusion、WeightedFusion、CascadeFusion），业务代码只依赖抽象接口，不依赖具体实现。当需要新增算法时，只需新增一个策略类，无需修改已有代码。

### 好处

- 算理系统中的算法变体，使算法可以独立于使用它的客户端而变化
- 消除冗长的条件判断语句，用多态替代分支
- 符合开闭原则，新增策略不影响已有代码
- 提高代码可读性和可维护性，每个策略类职责单一

---

## 二、工厂模式

### 代码示例

从 `knowledge/query/router/fusion.py` 中的 FusionFactory 提取：

```python
class FusionFactory:
    """策略工厂，负责创建具体策略实例"""
    
    _strategies = {
        "rrf": RRFFusion,
        "weighted": WeightedFusion,
        "cascade": CascadeFusion,
        "first_match": FirstMatchFusion,
    }
    
    @classmethod
    def create(cls, strategy: str, **kwargs) -> FusionStrategy:
        """根据策略名称创建策略实例"""
        if strategy not in cls._strategies:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        return cls._strategies[strategy](**kwargs)
    
    @classmethod
    def register(cls, name: str, strategy_cls: type) -> None:
        """动态注册新策略"""
        cls._strategies[name] = strategy_cls


# 使用示例
fusion = FusionFactory.create("rrf", k=60)
result = fusion.fuse(multiple_results)
```

从 `infra/model/qwen_factory.py` 中提取的模型工厂：

```python
class QwenFactory:
    """千问大模型工厂，支持场景路由"""
    
    def __init__(self, scenario: str = "default", **kwargs):
        # 根据场景选择不同的模型配置
        config_key_map = {
            "default": "chat_model_name",
            "router": "chat_model_name_router",
        }
        config_key = config_key_map.get(scenario, "chat_model_name")
        self.model = kwargs.get("model") or config.get(config_key)
        ...


# 使用示例
default_llm = QwenFactory(scenario="default")
router_llm = QwenFactory(scenario="router")
```

### 解决的问题

当对象创建逻辑复杂，涉及条件判断、配置读取、依赖组装时，如果在客户端直接使用构造函数创建对象，会导致客户端代码与创建逻辑耦合，难以测试和扩展。工厂模式将对象创建逻辑封装在工厂类中，客户端只需知道工厂接口和参数，无需了解创建细节。

### 原理

工厂模式的核心原理是"单一职责原则"——将对象创建职责从业务逻辑中分离出来。工厂类维护一个创建映射表（_strategies），将字符串标识符映射到具体类，通过统一接口（create方法）返回抽象类型（FusionStrategy）。客户端依赖抽象而非具体实现，创建逻辑集中管理。

### 好处

- 集中管理对象创建逻辑，避免创建代码分散在多个地方
- 降低客户端与具体类的耦合，客户端只需知道工厂和参数
- 便于扩展，新增产品类型只需更新工厂映射表
- 提高代码可测试性，可轻松 Mock 工厂返回测试对象

---

## 三、组合模式

### 代码示例

从 `knowledge/bridges/composite_cleaner.py` 中提取：

```python
class BaseCleaner:
    """清洗器基类"""
    def __init__(self, name: str):
        self._name = name
    
    def clean(self, text: str, **kwargs) -> str:
        return text


class CompositeCleaner(BaseCleaner):
    """组合清洗器，将多个清洗器组合成树形结构"""
    
    def __init__(self, cleaners: List[BaseCleaner], name: str = "composite"):
        super().__init__(name)
        self._cleaners = cleaners
    
    def clean(self, text: str, doc_type=None, level=None, metadata=None) -> str:
        result = text
        for cleaner in self._cleaners:
            try:
                result = cleaner.clean(result, doc_type, level, metadata)
            except Exception as e:
                self._logger.warning(f"Cleaner {cleaner._name} failed: {e}")
                continue
        return result
    
    def add_cleaner(self, cleaner: BaseCleaner) -> None:
        self._cleaners.append(cleaner)
    
    def remove_cleaner(self, name: str) -> bool:
        for i, cleaner in enumerate(self._cleaners):
            if cleaner._name == name:
                self._cleaners.pop(i)
                return True
        return False


# 使用示例：组合多个清洗器
composite = CompositeCleaner([
    HtmlCleaner(preserve_links=False),
    PrivacyCleaner(mask_ip=False),
    NoiseCleaner(),
    DuplicateCleaner(),
])
cleaned_text = composite.clean(raw_text)
```

### 解决的问题

当需要处理树形结构或层次结构的数据，且希望统一对待单个对象和组合对象时，如果分别编写处理单个对象和组合对象的代码，会导致客户端需要判断对象类型，增加复杂度。组合模式让客户端可以一致地处理单个对象和组合对象。

### 原理

组合模式的核心原理是"递归组合"——组合对象包含子对象列表，子对象可以是单个对象或另一个组合对象，形成树形结构。组合对象和单个对象实现相同接口（clean方法），客户端无需区分对象类型，递归调用自动处理整个树结构。

### 好处

- 统一对待单个对象和组合对象，客户端代码简洁
- 易于扩展新的组件类型，符合开闭原则
- 简化树形结构的遍历和处理逻辑
- 提供灵活的数据结构，可动态添加、删除组件

---

## 四、代理模式

### 代码示例

从 `infra/model/registry.py` 中的 ModelWrapper 提取：

```python
class ModelWrapper:
    """模型包装器，代理实际模型调用，添加熔断、重试、降级功能"""
    
    def __init__(self, registry, role, profile_name, fallback_chain, resilience):
        self._registry = registry
        self._role = role
        self._profile_name = profile_name
        self._fallback_chain = fallback_chain
        self._retry_strategy = RetryStrategy(resilience.get("retry", {}))
    
    def invoke(self, messages: List[dict], **kwargs) -> Any:
        """代理调用方法，添加额外功能"""
        profiles_to_try = [self._profile_name] + self._fallback_chain
        last_error = None
        
        for i, profile_name in enumerate(profiles_to_try):
            # 获取熔断器
            breaker = self._registry._get_circuit_breaker(f"{self._role}:{profile_name}")
            
            # 检查熔断器状态
            if breaker.is_open():
                self._logger.warning(f"Circuit breaker open for {profile_name}")
                continue
            
            try:
                # 调用真实对象
                provider = self._registry._get_provider(profile_name)
                result = provider.invoke(messages, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                last_error = e
                breaker.record_failure()
                self._logger.error(f"Model call failed: {e}")
                
                # 降级到下一个配置
                if i < len(profiles_to_try) - 1:
                    self._logger.info("Falling back to next profile")
        
        raise last_error or RuntimeError("No available model")


# 使用示例
registry = ModelRegistry()
model = registry.get_model("chat")  # 返回 ModelWrapper 代理
result = model.invoke(messages)     # 通过代理调用真实模型
```

### 解决的问题

当需要在访问对象时添加额外控制逻辑（如权限检查、缓存、日志、熔断、重试），但又不希望修改原始对象的代码时，如果直接在业务代码中添加这些逻辑，会导致业务逻辑与控制逻辑耦合。代理模式通过代理对象控制对原始对象的访问，在不修改原始对象的情况下扩展功能。

### 原理

代理模式的核心原理是"控制访问"——代理对象和真实对象实现相同接口（invoke方法），代理对象持有真实对象的引用，在调用真实对象前后添加额外逻辑（熔断检查、异常记录、降级切换）。客户端与代理交互，代理负责转发请求和控制访问。

### 好处

- 在不修改原始对象的情况下扩展功能，符合开闭原则
- 分离业务逻辑和控制逻辑，职责清晰
- 可添加多种控制功能：缓存、权限、日志、熔断、重试
- 客户端无感知，使用方式与原始对象一致

---

## 五、建造者模式

### 代码示例

从 `composition/factory.py` 中的 build_run_context 提取：

```python
class RunContext:
    """运行时上下文容器"""
    def __init__(self, request, rag, memory, tools, models, policy, privacy, ...):
        self.request = request
        self.rag = rag
        self.memory = memory
        self.tools = tools
        self.models = models
        self.policy = policy
        self.privacy = privacy
        ...


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
    """建造者函数，分步骤组装复杂对象"""
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
    """测试场景的建造者"""
    request = RequestContext(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        trace_id=trace_id,
        channel=channel
    )
    return build_run_context(request=request, **kwargs)


# 使用示例
context = build_run_context(
    request=request,
    rag=rag_adapter,
    memory=memory_adapter,
    tools=tool_adapter,
    models=model_registry,
    policy=policy_adapter,
)
```

### 解决的问题

当构建复杂对象涉及多个组成部分，且这些部分有默认值、可选参数、依赖关系时，如果直接使用构造函数创建对象，会导致参数列表过长、可选参数难以处理、构造逻辑分散。建造者模式将构建过程封装在专门的建造者函数中，提供清晰的构建接口和默认值处理。

### 原理

建造者模式的核心原理是"分步构建"——将复杂对象的构建过程分解为多个步骤，每个步骤设置对象的一个或多个属性。建造者函数接受可选参数，为未提供的参数提供默认值（如 FakeModelPort、FakePolicyPort），最终返回完整构建的对象。不同场景可以使用不同的建造者函数（build_run_context、build_test_context）。

### 好处

- 封装复杂构建逻辑，客户端无需了解构建细节
- 处理可选参数和默认值，参数列表清晰
- 支持多种构建场景，通过不同建造者函数定制
- 提高代码可读性，构建过程一目了然

---

## 六、装饰器模式

### 代码示例

从 `infra/model/qwen_factory.py` 中的 bind_tools 提取：

```python
class QwenFactory:
    """千问大模型工厂，支持工具绑定装饰"""
    
    def bind_tools(
        self,
        tools: List[Any],
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> "QwenFactory":
        """绑定工具到模型，返回增强后的新实例"""
        from langchain_core.utils.function_calling import convert_to_openai_function
        
        formatted_tools = []
        for tool in tools:
            if hasattr(tool, "args_schema"):
                formatted_tools.append(convert_to_openai_function(tool))
            else:
                formatted_tools.append(tool)
        
        # 创建新实例并添加工具能力
        new_instance = QwenFactory(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            ...
        )
        
        new_instance._bound_tools = [
            {"type": "function", "function": tool}
            for tool in formatted_tools
        ]
        new_instance._tool_choice = tool_choice
        
        return new_instance
    
    def _generate(self, messages, **kwargs):
        params = self._build_params(messages, **kwargs)
        
        # 添加绑定的工具（装饰功能）
        if hasattr(self, '_bound_tools'):
            params["tools"] = self._bound_tools
        if hasattr(self, '_tool_choice'):
            params["tool_choice"] = self._tool_choice
        
        response = self.client.chat.completions.create(**params)
        ...


# 使用示例
llm = QwenFactory()
llm_with_tools = llm.bind_tools([search_tool, calculator_tool])
response = llm_with_tools.invoke(messages)  # 使用增强后的模型
```

### 解决的问题

当需要在不修改原始对象代码的情况下动态添加功能，且希望保持原始对象不变时，如果使用继承会导致类爆炸，如果直接修改原始对象会破坏开闭原则。装饰器模式通过创建装饰对象包装原始对象，在不修改原始对象的情况下动态添加功能。

### 原理

装饰器模式的核心原理是"对象包装"——装饰器类和被装饰对象实现相同接口，装饰器持有被装饰对象的引用，在调用被装饰对象的方法前后添加额外功能。本项目中的 bind_tools 方法返回新实例而非修改原实例，实现非侵入式功能增强，原始对象保持不变，可多次装饰。

### 好处

- 动态添加功能，比继承更灵活
- 不修改原始对象，符合开闭原则
- 可组合多个装饰器，叠加多种功能
- 装饰过程透明，客户端使用方式不变

---

## 七、注册表模式

### 代码示例

从 `infra/model/registry.py` 中的 ModelRegistry 提取：

```python
class ModelRegistry:
    """模型注册表，集中管理模型配置和实例"""
    
    def __init__(self, config_path: str = "config/models.yml"):
        self._profiles: Dict[str, Profile] = {}
        self._roles: Dict[str, Role] = {}
        self._providers: Dict[str, Any] = {}
        self._instances: Dict[str, Any] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        self._load_config()
    
    def _load_config(self) -> None:
        """从配置文件加载注册信息"""
        with open(self._config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        for name, profile_data in config.get("profiles", {}).items():
            self._profiles[name] = Profile(
                provider=profile_data.get("provider", ""),
                model_name=profile_data.get("model_name", ""),
                ...
            )
        
        for name, role_data in config.get("roles", {}).items():
            self._roles[name] = Role(
                profile=role_data.get("profile", ""),
                fallback_chain=role_data.get("fallback_chain", []),
                ...
            )
    
    def get_model(self, role: str) -> Any:
        """根据角色获取模型实例"""
        role_config = self._roles.get(role)
        if not role_config:
            raise ValueError(f"Role not found: {role}")
        
        return ModelWrapper(
            registry=self,
            role=role,
            profile_name=role_config.profile,
            fallback_chain=role_config.fallback_chain,
            ...
        )
    
    def get_model_info(self, role: str) -> ModelInfo:
        """获取模型元信息"""
        ...
    
    def invalidate_cache(self, role: Optional[str] = None) -> None:
        """清除缓存"""
        if role:
            self._instances.pop(role, None)
        else:
            self._instances.clear()


# 使用示例
registry = ModelRegistry("config/models.yml")
chat_model = registry.get_model("chat")
embedding_model = registry.get_model("embedding")
```

### 解决的问题

当系统中有多个对象需要根据名称或标识符查找，且这些对象有配置依赖、需要缓存、需要统一管理时，如果在各处分散创建和查找逻辑，会导致代码重复、难以维护。注册表模式提供统一的查找和管理机制，集中管理对象实例。

### 原理

注册表模式的核心原理是"集中管理"——注册表类维护一个或多个字典（_profiles、_roles、_providers），将标识符映射到对象或配置。注册表提供统一的查找接口（get_model），内部处理配置加载、实例创建、缓存管理、熔断器管理等逻辑。客户端通过注册表获取对象，无需了解创建细节。

### 好处

- 集中管理对象配置和实例，避免分散创建
- 提供统一查找接口，简化客户端代码
- 支持缓存机制，避免重复创建开销
- 便于扩展，新增对象类型只需更新注册表

---

## 八、模板方法模式

### 代码示例

从 `knowledge/bridges/cleaners/base_cleaners.py` 和 `composite_cleaner.py` 中提取：

```python
class BaseCleaner:
    """清洗器基类，定义清洗流程骨架"""
    
    def __init__(self, name: str):
        self._name = name
        self._logger = logging.getLogger(f"cleaner.{name}")
    
    def clean(self, text: str, doc_type=None, level=None, metadata=None) -> str:
        """模板方法，定义清洗流程骨架"""
        if not text:
            return text
        
        # 前置处理（子类可选重写）
        text = self._pre_process(text, metadata)
        
        # 核心清洗逻辑（子类必须实现）
        text = self._do_clean(text, doc_type, level, metadata)
        
        # 后置处理（子类可选重写）
        text = self._post_process(text, metadata)
        
        return text
    
    def _pre_process(self, text: str, metadata=None) -> str:
        """前置处理钩子方法，默认不做处理"""
        return text
    
    def _do_clean(self, text: str, doc_type, level, metadata) -> str:
        """核心清洗方法，子类实现具体清洗逻辑"""
        return text
    
    def _post_process(self, text: str, metadata=None) -> str:
        """后置处理钩子方法，默认不做处理"""
        return text


class HtmlCleaner(BaseCleaner):
    """HTML清洗器"""
    
    def __init__(self, preserve_links: bool = False):
        super().__init__("html")
        self._preserve_links = preserve_links
    
    def _do_clean(self, text, doc_type, level, metadata) -> str:
        # 具体的HTML清洗实现
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()


class PrivacyCleaner(BaseCleaner):
    """隐私数据清洗器"""
    
    def __init__(self, mask_email: bool = True, mask_phone: bool = True):
        super().__init__("privacy")
        self._mask_email = mask_email
        self._mask_phone = mask_phone
    
    def _do_clean(self, text, doc_type, level, metadata) -> str:
        # 具体的隐私清洗实现
        import re
        if self._mask_email:
            text = re.sub(r'[\w\.-]+@[\w\.-]+', '***@***', text)
        if self._mask_phone:
            text = re.sub(r'1[3-9]\d{9}', '***********', text)
        return text


# 使用示例
html_cleaner = HtmlCleaner(preserve_links=False)
privacy_cleaner = PrivacyCleaner(mask_email=True, mask_phone=True)

text1 = html_cleaner.clean(raw_html)
text2 = privacy_cleaner.clean(text1)
```

### 解决的问题

当多个类有相似的算法流程，只有部分步骤不同时，如果让每个类都实现完整流程，会导致代码重复、难以维护。模板方法模式在基类中定义算法骨架，将不变的部分实现，将变化的部分延迟到子类实现。

### 原理

模板方法模式的核心原理是"控制反转"——基类定义算法骨架（clean方法），调用抽象方法或钩子方法（_do_clean、_pre_process、_post_process），具体实现延迟到子类。算法流程由基类控制，具体步骤由子类提供，实现"不要调用我，让我调用你"的控制反转。

### 好处

- 复用算法骨架，避免代码重复
- 集中控制算法流程，便于维护
- 子类只需实现差异部分，降低开发成本
- 符合开闭原则，扩展新实现不影响已有代码

---

## 九、Port-Adapter架构模式

### 代码示例

从项目结构中提取：

**Port定义**（抽象接口）：

```python
from typing import Protocol

class RelationalPort(Protocol):
    """关系数据库Port，定义统一接口"""
    
    async def execute(self, sql: str, params: tuple = ()) -> None:
        """执行SQL"""
        ...
    
    async def insert(self, table: str, data: dict) -> int:
        """插入数据"""
        ...
    
    async def select_one(self, table: str, where: dict) -> Optional[dict]:
        """查询单条"""
        ...


class CachePort(Protocol):
    """缓存Port，定义统一接口"""
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        ...
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """设置缓存"""
        ...
    
    async def delete(self, key: str) -> None:
        """删除缓存"""
        ...
```

**Adapter实现**：

```python
# SQLite适配器
class AsyncSQLiteRelationalAdapter:
    """SQLite关系数据库适配器"""
    
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._pool = None
    
    async def execute(self, sql: str, params: tuple = ()) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(sql, params)
    
    async def insert(self, table: str, data: dict) -> int:
        async with self._pool.acquire() as conn:
            cursor = await conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                tuple(data.values())
            )
            return cursor.lastrowid
    
    async def select_one(self, table: str, where: dict) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            cursor = await conn.execute(
                f"SELECT * FROM {table} WHERE {conditions}",
                tuple(where.values())
            )
            row = await cursor.fetchone()
            return dict(row) if row else None


# 内存缓存适配器
class AsyncMemoryCacheAdapter:
    """内存缓存适配器"""
    
    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._default_ttl = default_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        value, expire_time = self._cache[key]
        if time.time() > expire_time:
            del self._cache[key]
            return None
        return value
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        expire_time = time.time() + (ttl or self._default_ttl)
        self._cache[key] = (value, expire_time)
    
    async def delete(self, key: str) -> None:
        self._cache.pop(key, None)
```

**业务逻辑**（依赖Port，不依赖Adapter）：

```python
class SessionManager:
    """会话管理器，依赖抽象Port"""
    
    def __init__(self, db: RelationalPort, cache: CachePort):
        self._db = db      # 依赖抽象
        self._cache = cache
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        # 先查缓存
        cached = await self._cache.get(f"session:{session_id}")
        if cached:
            return Session.from_dict(cached)
        
        # 缓存未命中，查数据库
        data = await self._db.select_one("sessions", {"id": session_id})
        if data:
            await self._cache.set(f"session:{session_id}", data, ttl=3600)
            return Session.from_dict(data)
        
        return None


# 组装：由工厂注入具体Adapter
session_manager = SessionManager(
    db=AsyncSQLiteRelationalAdapter("data.db"),      # 注入SQLite适配器
    cache=AsyncMemoryCacheAdapter(default_ttl=3600)  # 注入内存缓存适配器
)

# 或在生产环境使用不同适配器
session_manager_prod = SessionManager(
    db=PostgreSQLAdapter(host="localhost", database="prod"),
    cache=RedisCacheAdapter(host="localhost", port=6379)
)
```

### 解决的问题

当业务逻辑依赖外部系统（数据库、缓存、消息队列、第三方API）时，如果直接调用外部系统的API，会导致业务逻辑与外部系统耦合，难以测试、难以替换外部系统。Port-Adapter架构将业务逻辑与外部依赖分离，业务逻辑依赖抽象Port，具体实现由Adapter提供，实现依赖倒置。

### 原理

Port-Adapter架构的核心原理是"依赖倒置原则"——高层模块（业务逻辑）不应依赖低层模块（外部系统），两者都应依赖抽象（Port）。Port定义业务需要的接口（Protocol），Adapter实现Port接口，将外部系统的API适配为Port接口。业务逻辑只依赖Port，由工厂或依赖注入框架在运行时注入具体Adapter。

### 好处

- 业务逻辑与技术实现分离，符合单一职责原则
- 依赖倒置，业务逻辑不依赖具体技术，易于测试
- 可替换外部系统，切换数据库或缓存无需修改业务代码
- 支持多种实现，开发环境用SQLite，生产环境用PostgreSQL
- 提高代码可维护性，技术变更影响范围小

---

## 十、管道模式

### 代码示例

从 `knowledge/pipeline/ingest/adapters/simplified_adapter.py` 中提取：

```python
class SimplifiedIngestPipeline:
    """文档摄入管道，串联多个处理步骤"""
    
    def __init__(
        self,
        parser: ParserPort,
        cleaner: CleanerPort,
        chunker: ChunkerPort,
        indexer: IndexerPort
    ):
        self._parser = parser
        self._cleaner = cleaner
        self._chunker = chunker
        self._indexer = indexer
    
    async def process(self, document: Document) -> ProcessingResult:
        """管道处理流程：解析 → 清洗 → 分块 → 索引"""
        # 步骤1：解析文档
        parsed = await self._parser.parse(document.content, document.format)
        
        # 步骤2：清洗内容
        cleaned = await self._cleaner.clean(
            parsed.text,
            doc_type=document.type,
            level=CleaningLevel.STANDARD
        )
        
        # 步骤3：分块处理
        chunks = await self._chunker.chunk(
            cleaned,
            strategy=ChunkingStrategy.SEMANTIC,
            max_chunk_size=512
        )
        
        # 步骤4：索引存储
        indexed = await self._indexer.index(
            chunks,
            collection=document.collection,
            metadata=document.metadata
        )
        
        return ProcessingResult(
            document_id=document.id,
            chunks_count=len(chunks),
            indexed_ids=indexed.ids
        )


# 使用示例
pipeline = SimplifiedIngestPipeline(
    parser=LayoutParser(),
    cleaner=CompositeCleaner([HtmlCleaner(), PrivacyCleaner()]),
    chunker=SemanticChunker(),
    indexer=VectorIndexer()
)

result = await pipeline.process(document)
```

### 解决的问题

当处理流程包含多个连续步骤，且步骤之间需要传递中间结果时，如果将所有步骤写在一个函数中，会导致代码冗长、难以复用单个步骤。管道模式将流程分解为独立的步骤，每个步骤专注单一职责，步骤之间通过管道传递数据。

### 原理

管道模式的核心原理是"分而治之"——将复杂流程分解为多个简单步骤，每个步骤是一个独立的处理单元（函数或对象）。步骤接收输入，处理后产生输出，输出成为下一个步骤的输入。通过组合不同步骤，可以灵活构建不同流程，步骤可独立测试和复用。

### 好处

- 将复杂流程分解为简单步骤，每个步骤职责单一
- 步骤可独立测试、独立复用，提高可维护性
- 支持灵活组合，通过配置构建不同流程
- 易于扩展，新增步骤不影响已有步骤

---

## 总结

本项目综合运用了多种设计模式，体现了良好的架构设计：

- **策略模式**：算法可插拔，支持多种融合策略、清洗策略等
- **工厂模式**：集中管理对象创建，降低客户端耦合
- **组合模式**：统一处理单个对象和组合对象，简化树形结构处理
- **代理模式**：控制对象访问，添加熔断、重试、降级等功能
- **建造者模式**：分步骤构建复杂对象，处理可选参数和默认值
- **装饰器模式**：动态添加功能，不修改原始对象
- **注册表模式**：集中管理对象配置和实例，提供统一查找接口
- **模板方法模式**：定义算法骨架，子类实现差异部分
- **Port-Adapter架构**：分离业务逻辑与技术实现，实现依赖倒置
- **管道模式**：串联多个处理步骤，支持灵活组合

这些设计模式的综合应用，使项目具备高可扩展性、高可维护性、高可测试性，充分体现了企业级Python应用的设计最佳实践。
