import os
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

import torch
from transformers import AutoConfig
from transformers import PretrainedConfig


@dataclass
class SoulXPodcastLLMConfig:
    architectures: list[str] = field(default_factory=lambda: ["Qwen3ForCausalLM"])
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151675  # speech eos
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 6144
    max_position_embeddings: int = 40960
    max_window_layers: int = 28
    model_type: str = "qwen3"
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-06
    rope_scaling: dict | None = None
    rope_theta: float = 1000000.0
    sliding_window: int = 32768
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.52.3"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 159488  # text_vocab_size + speech_vocab_size + 2 (eos and task_id)
    lm_head_bias: bool = False
    qkv_bias: bool = False
    fp16_flow: bool = False
    speech_token_offset: int = 152927

    @classmethod
    def from_initial_and_json(
        cls, 
        initial_values: Dict[str, Any] = None, 
        json_file: Optional[str] = None
    ):
        """
        Create an instance from initial values and JSON data.
        
        Args:
            initial_values: Dictionary of initial values (highest priority)
            json_file: Path to JSON file

        Returns:
            SoulXPodcastLLMConfig instance
        """
        # Merge all data sources
        merged_data = {}
        
        # 1. Load from JSON file first (lowest priority)
        if json_file and os.path.exists(json_file):
            file_data = cls._load_json_file(json_file)
            merged_data.update(file_data)
        
         # 2. Overwrite with initial values (highest priority)
        if initial_values:
            merged_data.update(initial_values)
        
        # Filter dataclass fields
        valid_fields = {f.name for f in fields(cls)}
        init_data = {k: v for k, v in merged_data.items() if k in valid_fields}
        
        return cls(**init_data)

    @staticmethod
    def _load_json_file(file_path: str) -> Dict[str, Any]:
        """Load data from a JSON file"""
        path = Path(file_path)
        if not path.exists():
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

class AutoPretrainedConfig(PretrainedConfig):
    model_type = "qwen3"
    
    def __init__(self, **kwargs):
        # Filter out non-configuration parameters
        config_kwargs = {k: v for k, v in kwargs.items() 
                        if not k.startswith('_') and k != 'self'}
        super().__init__(**config_kwargs)
    
    @classmethod
    def from_dataclass(cls, dataclass_config):
        """Automatically create configuration from any dataclass"""
        if not is_dataclass(dataclass_config):
            raise ValueError("Input must be a dataclass instance")
        
        dataclass_dict = asdict(dataclass_config)
        return cls(**dataclass_dict)


@dataclass
class SamplingParams:
    temperature: float = 0.6
    repetition_penalty: float = 1.25
    top_k: int = 100
    top_p: float = 0.9
    min_tokens: int = 8
    max_tokens: int = 3000
    stop_token_ids: list[int] = field(default_factory=lambda: [151675])
    # RasSampler parameters
    use_ras: bool = True
    win_size: int = 25
    tau_r: float = 0.2


@dataclass
class Config:
    model: str
    max_model_len: int = 8192  # 15s prompt + 30s generated audio for 25hz audio tokenizer
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: SoulXPodcastLLMConfig | AutoConfig = field(default_factory=SoulXPodcastLLMConfig)
    eos: int = -1
    llm_engine: str = "hf" # support hf, nano-vllm
    max_turn_size: int = 10
    turn_tokens_threshold: int = 6192
    
    prompt_context: int = 2 # default to 2 for two-speaker podcast;
    history_context: int = 2
    history_text_context: int = 2
    
    def __post_init__(self):
        assert os.path.isdir(self.model)

        max_pos = getattr(self.hf_config, "max_position_embeddings", 8192)
        self.max_model_len = min(self.max_model_len, max_pos)