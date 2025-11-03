"""
API Configuration Management
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIConfig:
    """API配置类"""

    # 模型配置
    model_path: str = os.getenv(
        "MODEL_PATH",
        "pretrained_models/SoulX-Podcast-1.7B"
    )
    llm_engine: str = os.getenv("LLM_ENGINE", "hf")  # hf or vllm

    def validate_llm_engine(self):
        """验证LLM引擎配置"""
        if self.llm_engine not in ["hf", "vllm"]:
            raise ValueError(f"Invalid llm_engine: {self.llm_engine}. Must be 'hf' or 'vllm'")

        # 如果选择vllm，检查是否安装
        if self.llm_engine == "vllm":
            try:
                import vllm
            except ImportError:
                import logging
                logging.warning("vLLM not installed, falling back to HuggingFace engine")
                self.llm_engine = "hf"
    fp16_flow: bool = os.getenv("FP16_FLOW", "false").lower() == "true"

    # 服务配置
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"

    # 文件配置
    temp_dir: Path = Path("api/temp")
    output_dir: Path = Path("api/outputs")
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    file_cleanup_minutes: int = 30  # 文件过期时间（分钟）

    # 并发控制
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", "2"))

    # 默认生成参数
    default_seed: int = 1988
    default_temperature: float = 0.6
    default_top_k: int = 100
    default_top_p: float = 0.9

    def __post_init__(self):
        """确保目录存在并验证配置"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validate_llm_engine()


# 全局配置实例
config = APIConfig()
