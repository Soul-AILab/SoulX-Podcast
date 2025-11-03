"""
测试单例模式是否正常工作
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.service import get_service

def test_singleton():
    """测试单例模式"""
    print("=== 测试单例模式 ===")

    # 第一次获取服务
    print("\n第1次调用 get_service()...")
    service1 = get_service()
    print(f"Service1 ID: {id(service1)}")
    print(f"Service1 Model: {id(service1.model) if hasattr(service1, 'model') else 'Not loaded'}")

    # 第二次获取服务
    print("\n第2次调用 get_service()...")
    service2 = get_service()
    print(f"Service2 ID: {id(service2)}")
    print(f"Service2 Model: {id(service2.model) if hasattr(service2, 'model') else 'Not loaded'}")

    # 第三次获取服务
    print("\n第3次调用 get_service()...")
    service3 = get_service()
    print(f"Service3 ID: {id(service3)}")
    print(f"Service3 Model: {id(service3.model) if hasattr(service3, 'model') else 'Not loaded'}")

    # 验证是否是同一个实例
    print("\n=== 验证结果 ===")
    print(f"service1 is service2: {service1 is service2}")
    print(f"service2 is service3: {service2 is service3}")
    print(f"service1.model is service2.model: {service1.model is service2.model if hasattr(service1, 'model') and hasattr(service2, 'model') else 'N/A'}")

    # 验证模型组件是否相同
    if hasattr(service1, 'model'):
        print(f"\n=== 模型组件ID ===")
        print(f"audio_tokenizer: {id(service1.model.audio_tokenizer)}")
        print(f"llm: {id(service1.model.llm)}")
        print(f"flow: {id(service1.model.flow)}")
        print(f"hift: {id(service1.model.hift)}")

if __name__ == "__main__":
    test_singleton()