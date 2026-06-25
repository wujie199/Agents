"""L1 基础设施 Port 装配工厂。

构建 ConfigPort、SecretPort、PrivacyPort、IdentityPort、PolicyPort、ObservabilityPort。
"""

from dataclasses import dataclass

from agent_platform.infrastructure.config.adapter import ConfigPortAdapter
from agent_platform.infrastructure.secret.adapter import SecretPortAdapter
from agent_platform.infrastructure.privacy.adapter import PrivacyPortAdapter
from agent_platform.infrastructure.identity.adapter import IdentityPortAdapter
from agent_platform.infrastructure.policy.adapter import PolicyPortAdapter
from agent_platform.infrastructure.observability.adapter import ObservabilityPortAdapter


@dataclass
class InfrastructurePorts:
    """L1 横切关注点端口集合。"""
    config: ConfigPortAdapter
    secret: SecretPortAdapter
    privacy: PrivacyPortAdapter
    identity: IdentityPortAdapter
    policy: PolicyPortAdapter
    observability: ObservabilityPortAdapter


def build_infrastructure_ports(
    config_dir: str = "config",
    service_name: str = "agents",
) -> InfrastructurePorts:
    """构建所有 L1 横切关注点端口。

    Args:
        config_dir: 配置文件目录
        service_name: 服务名称（用于 observability）
    """
    return InfrastructurePorts(
        config=ConfigPortAdapter(config_dir=config_dir),
        secret=SecretPortAdapter(),
        privacy=PrivacyPortAdapter(),
        identity=IdentityPortAdapter(),
        policy=PolicyPortAdapter(config_path=f"{config_dir}/concurrency.yml"),
        observability=ObservabilityPortAdapter(service_name=service_name),
    )
