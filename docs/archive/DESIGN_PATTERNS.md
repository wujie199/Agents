# ports 与 infrastructure 的设计模式

## 核心模式：依赖倒置 + 适配器

### 1. 依赖倒置原则（DIP）

**定义**：高层模块不依赖低层模块，两者都依赖抽象。

```
传统方式（错误）：
┌─────────────┐
│ Agent       │ ──依赖──> Redis SDK
└─────────────┘
     ↓
  无法替换 Redis


依赖倒置（正确）：
┌─────────────┐
│ Agent       │ ──依赖──> CachePort (抽象)
└─────────────┘                    ↑
                                   │ 实现
                         ┌─────────┴─────────┐
                         │ RedisCacheAdapter │
                         └───────────────────┘
                                   │
                                   ↓
                              Redis SDK
```

**代码体现**：

```python
# Agent 只依赖抽象（Protocol）
async def worker(ctx: RunContext):
    data = await ctx.extra["cache"].get("key")  # 依赖 CachePort
    # 不关心底层是 Redis 还是 Memory

# 抽象定义
class CachePort(Protocol):
    def get(self, key: str) -> Optional[Any]: ...

# 具体实现
class RedisCacheAdapter:  # 不需要继承 Protocol
    def get(self, key: str) -> Optional[Any]:
        return self._redis.get(key)  # 适配 Redis SDK
```

---

### 2. 适配器模式（Adapter Pattern）

**定义**：将一个类的接口转换成客户期望的另一个接口。

```
┌──────────────────────────────────────────────┐
│              CachePort (Target)              │
│  + get(key) -> Any                           │
│  + set(key, value, ttl) -> None              │
└──────────────────────────────────────────────┘
                      ↑
                      │ 实现
┌──────────────────────────────────────────────┐
│         RedisCacheAdapter (Adapter)          │
├──────────────────────────────────────────────┤
│  - client: redis.Redis (Adaptee)             │
├──────────────────────────────────────────────┤
│  + get(key):                                 │
│      return self._client.get(key)  ← 适配    │
│  + set(key, value, ttl):                     │
│      self._client.setex(key, ttl, value)     │
└──────────────────────────────────────────────┘
                      │
                      ↓
┌──────────────────────────────────────────────┐
│           redis.Redis (Adaptee)               │
│  + get(name) -> str                          │
│  + setex(name, time, value) -> bool          │
└──────────────────────────────────────────────┘
```

**作用**：
- 将 Redis SDK 的接口适配到 CachePort 接口
- 将 Chroma SDK 的接口适配到 VectorPort 接口
- 将 Neo4j SDK 的接口适配到 GraphPort 接口

---

### 3. 工厂模式（Factory Pattern）

**定义**：创建对象的接口，让子类决定实例化哪个类。

```python
# composition/factory.py

def build_production_context(request: RequestContext) -> RunContext:
    """工厂方法：创建生产环境配置"""
    
    cache = EnterpriseRedisCacheAdapter(  # 创建具体实现
        host="redis.prod.com",
        pool_size=20
    )
    
    return RunContext(
        request=request,
        extra={"cache": cache}
    )


def build_development_context(request: RequestContext) -> RunContext:
    """工厂方法：创建开发环境配置"""
    
    cache = MemoryCacheAdapter()  # 创建不同实现
    
    return RunContext(
        request=request,
        extra={"cache": cache}
    )
```

**作用**：
- 决定使用哪个 Adapter
- 组装复杂的依赖关系
- 隔离创建逻辑

---

### 4. 策略模式（Strategy Pattern）

**定义**：定义算法族，让它们可以互相替换。

```python
# 不同策略（不同实现）

# 策略 1：Redis 缓存
cache = EnterpriseRedisCacheAdapter(host="redis.com")

# 策略 2：内存缓存
cache = MemoryCacheAdapter()

# 策略 3：无缓存
cache = NoOpCacheAdapter()

# 客户端代码不变
data = cache.get("key")  # 使用同一接口
```

**作用**：
- 通过配置切换不同实现
- 运行时可替换策略

---

## 不是装饰器模式

**装饰器模式**的定义：动态地给对象添加额外职责。

```python
# 这是装饰器模式（我们没用）
class LoggingCacheDecorator:
    def __init__(self, cache: CachePort):
        self._cache = cache  # 包装原对象
    
    def get(self, key: str):
        print(f"Getting {key}")  # 添加日志
        return self._cache.get(key)  # 调用原对象

# 使用
cache = RedisCacheAdapter()
cache = LoggingCacheDecorator(cache)  # 装饰
data = cache.get("key")  # 带日志的缓存
```

**为什么不用装饰器**：
- Port 层已经定义了接口，不需要再包装
- 横切关注点（日志、审计）在 Middleware 层处理
- 保持架构简洁

---

## 模式总结

| 模式 | 位置 | 作用 |
|------|------|------|
| **依赖倒置** | ports/ vs infrastructure/ | 解耦高层与低层 |
| **适配器** | infrastructure/adapters/ | 适配第三方 SDK |
| **工厂** | composition/factory.py | 创建和组装对象 |
| **策略** | 配置驱动 | 运行时切换实现 |

---

## 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    业务层 (Agent)                        │
│                      ↓ 依赖                             │
├─────────────────────────────────────────────────────────┤
│                  ports/ (抽象层)                         │
│            Protocol 接口定义 (依赖倒置)                  │
│                      ↑ 实现                             │
├─────────────────────────────────────────────────────────┤
│              infrastructure/ (实现层)                    │
│                                                         │
│  ┌────────────────┐    适配器模式                        │
│  │ RedisAdapter   │ ──适配──> Redis SDK                 │
│  │ ChromaAdapter  │ ──适配──> Chroma SDK                │
│  │ Neo4jAdapter   │ ──适配──> Neo4j SDK                 │
│  └────────────────┘                                     │
└─────────────────────────────────────────────────────────┘
                      ↑ 创建
┌─────────────────────────────────────────────────────────┐
│           composition/factory.py (工厂)                  │
│    build_production_context() → 创建生产配置             │
│    build_development_context() → 创建开发配置            │
└─────────────────────────────────────────────────────────┘
```

---

## 关键点

1. **依赖倒置**：Agent 依赖 `ports/`，不依赖 `infrastructure/`
2. **适配器**：每个 Adapter 适配一个第三方 SDK
3. **工厂**：工厂决定创建哪个 Adapter
4. **策略**：通过配置或环境选择不同实现

这是典型的 **Clean Architecture / Hexagonal Architecture** 的实现方式。
