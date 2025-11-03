"""
测试 vLLM 配置和引擎切换
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置 UTF-8 编码（Windows 兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 测试配置加载
print("=" * 60)
print("测试 API 配置")
print("=" * 60)

# 测试 HF 引擎
os.environ["LLM_ENGINE"] = "hf"
from api.config import APIConfig
config_hf = APIConfig()
print(f"\n[OK] HF 引擎配置: {config_hf.llm_engine}")

# 测试 vLLM 引擎
os.environ["LLM_ENGINE"] = "vllm"
# 重新导入以应用新的环境变量
import importlib
import api.config
importlib.reload(api.config)
from api.config import APIConfig
config_vllm = APIConfig()
print(f"[OK] vLLM 引擎配置: {config_vllm.llm_engine}")

# 检查 vLLM 是否安装
print("\n" + "=" * 60)
print("检查 vLLM 安装状态")
print("=" * 60)

try:
    import vllm
    print(f"\n[OK] vLLM 已安装 (版本: {vllm.__version__})")
    vllm_available = True
except ImportError:
    print("\n[WARN] vLLM 未安装")
    print("  安装命令: pip install vllm")
    vllm_available = False

# 检查 LLM 引擎模块
print("\n" + "=" * 60)
print("检查 LLM 引擎模块")
print("=" * 60)

from soulxpodcast.engine.llm_engine import SUPPORT_VLLM, HFLLMEngine
print(f"\n[OK] HFLLMEngine 可用")
print(f"[OK] SUPPORT_VLLM = {SUPPORT_VLLM}")

if SUPPORT_VLLM:
    from soulxpodcast.engine.llm_engine import VLLMEngine
    print(f"[OK] VLLMEngine 可用")
else:
    print("[WARN] VLLMEngine 不可用（vLLM 未安装）")

# 总结
print("\n" + "=" * 60)
print("配置总结")
print("=" * 60)

print(f"""
当前配置：
- 模型路径: {config_vllm.model_path}
- LLM 引擎: {config_vllm.llm_engine}
- FP16 Flow: {config_vllm.fp16_flow}
- 最大并发任务: {config_vllm.max_concurrent_tasks}

引擎支持：
- HuggingFace: [OK] 可用
- vLLM: {'[OK] 可用' if vllm_available else '[WARN] 不可用（需要安装）'}

推荐配置：
""")

if vllm_available:
    print("""
# 使用 vLLM 引擎（推荐生产环境）
LLM_ENGINE=vllm \\
MAX_CONCURRENT_TASKS=3 \\
python api/main.py
""")
else:
    print("""
# 使用 HuggingFace 引擎（默认）
LLM_ENGINE=hf \\
MAX_CONCURRENT_TASKS=2 \\
python api/main.py

# 安装 vLLM 以获得更好的性能：
pip install vllm
""")

print("=" * 60)
