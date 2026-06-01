"""
演示 Protocol 如何在代码中体现

Python 的 Protocol 是通过鸭子类型实现的：
- 不需要显式继承
- 只要方法签名匹配，就被认为是该 Protocol 的实现
"""

from typing import Protocol
from core.ports.privacy import PrivacyPort
from agent_platform.infrastructure.privacy.adapter import PrivacyPortAdapter
from core.composition.factory import FakePrivacyPort


def demo_protocol_check():
    """演示 Protocol 类型检查"""
    
    # 1. PrivacyPortAdapter 没有继承 PrivacyPort
    #    但因为它实现了所有方法，所以"被视为" PrivacyPort
    
    adapter = PrivacyPortAdapter()
    fake = FakePrivacyPort()
    
    # 两者都可以赋值给 PrivacyPort 类型的变量
    privacy: PrivacyPort = adapter  # ✅ 类型检查通过
    privacy: PrivacyPort = fake     # ✅ 类型检查通过
    
    # 调用方式完全相同
    result1 = adapter.mask_text("手机13812345678")
    result2 = fake.mask_text("手机13812345678")
    
    print(f"Adapter 结果: {result1}")
    print(f"Fake 结果: {result2}")


def demo_usage_in_business():
    """演示业务代码中如何使用 Protocol"""
    
    # 业务函数的参数类型是 Protocol
    def process_sensitive_data(text: str, privacy: PrivacyPort) -> str:
        """
        参数类型是 PrivacyPort（接口）
        可以接受任何实现了该接口的对象
        """
        masked = privacy.mask_text(text)
        hashed = privacy.hash_for_audit(text)
        return f"masked: {masked}, hash: {hashed}"
    
    # 可以传入真实实现
    adapter = PrivacyPortAdapter()
    result1 = process_sensitive_data("手机13812345678", adapter)
    
    # 也可以传入 Fake 实现（测试用）
    fake = FakePrivacyPort()
    result2 = process_sensitive_data("手机13812345678", fake)
    
    print(f"真实实现: {result1}")
    print(f"Fake实现: {result2}")


def demo_method_signature_matching():
    """演示方法签名必须匹配"""
    
    # ✅ 正确：实现所有 Protocol 定义的方法
    class CorrectImplementation:
        def mask_text(self, text: str, policy: str = None) -> str:
            return text
        
        def redact_for_storage(self, record: dict) -> dict:
            return record
        
        def redact_for_llm(self, messages: list, policy: str = None) -> list:
            return messages
        
        def hash_for_audit(self, value: str) -> str:
            return "hash"
        
        def classify_sensitivity(self, text: str):
            return "public"
    
    # 类型检查通过
    privacy: PrivacyPort = CorrectImplementation()
    
    # ❌ 错误：缺少方法
    # class WrongImplementation:
    #     def mask_text(self, text: str) -> str:
    #         return text
    # 
    # # 类型检查失败（运行时可能不报错，但静态检查会报警）
    # privacy: PrivacyPort = WrongImplementation()  # ⚠️ 警告


if __name__ == "__main__":
    print("=" * 50)
    print("1. Protocol 类型检查演示")
    print("=" * 50)
    demo_protocol_check()
    
    print("\n" + "=" * 50)
    print("2. 业务代码使用 Protocol 演示")
    print("=" * 50)
    demo_usage_in_business()
    
    print("\n" + "=" * 50)
    print("3. 方法签名匹配演示")
    print("=" * 50)
    demo_method_signature_matching()
    print("✅ 正确实现可以通过类型检查")
