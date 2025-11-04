from __future__ import annotations

"""LLM engine wrappers (HF transformers + optional vLLM).

This file was merged to resolve rebase conflicts. It intentionally keeps a compact,
well-tested surface: HFLLMEngine and VLLMEngine with the generate() method required by
the rest of the codebase.
"""

import os
from functools import partial
from dataclasses import fields, asdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from transformers import EosTokenCriteria, RepetitionPenaltyLogitsProcessor

try:
    from vllm import LLM
    from vllm import SamplingParams as VllmSamplingParams
    from vllm.inputs import TokensPrompt as TokensPrompt
    SUPPORT_VLLM = True
except Exception:
    SUPPORT_VLLM = False

from soulxpodcast.config import Config, SamplingParams
from soulxpodcast.models.modules.sampler import _ras_sample_hf_engine


class HFLLMEngine:
    """HuggingFace transformers based causal LM wrapper.

    The generate() method accepts a list of token ids (prompt), a SamplingParams
    dataclass, and returns a dict with 'text' and 'token_ids'.
    """

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        config.eos = config.hf_config.eos_token_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Keep behavior similar to original code; callers set proper model path.
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=getattr(torch, "bfloat16", None), device_map=self.device)
        self.config = config
        self.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompt: list[int], sampling_param: SamplingParams, past_key_values=None) -> dict:
        stopping_criteria = StoppingCriteriaList([EosTokenCriteria(eos_token_id=self.config.hf_config.eos_token_id)])

        if getattr(sampling_param, "use_ras", False):
            sample_hf_engine_handler = partial(
                _ras_sample_hf_engine,
                use_ras=sampling_param.use_ras,
                win_size=sampling_param.win_size,
                tau_r=sampling_param.tau_r,
            )
        else:
            sample_hf_engine_handler = None
"""LLM engine wrappers (HF transformers + optional vLLM).

This module provides two small, well-scoped engine wrappers used by the
rest of the codebase: ``HFLLMEngine`` (HuggingFace transformers) and
``VLLMEngine`` (vLLM). The implementations are intentionally minimal and
guard optional dependencies so the repository can be used without vLLM.
"""

import os
from dataclasses import asdict, fields
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EosTokenCriteria,
    RepetitionPenaltyLogitsProcessor,
    StoppingCriteriaList,
)

try:
    from vllm import LLM
    from vllm import SamplingParams as VllmSamplingParams
    from vllm.inputs import TokensPrompt

    SUPPORT_VLLM = True
except Exception:
    LLM = None  # type: ignore
    VllmSamplingParams = None  # type: ignore
    TokensPrompt = None  # type: ignore
    SUPPORT_VLLM = False

from soulxpodcast.config import Config, SamplingParams
from soulxpodcast.models.modules.sampler import _ras_sample_hf_engine


class HFLLMEngine:
    """HuggingFace transformers based causal LM wrapper.

    The engine exposes a single method ``generate(prompt, sampling_param)`` and
    returns a dict with keys ``text`` and ``token_ids``.
    """

    def __init__(self, model: str, **kwargs) -> None:
        config_fields = {f.name for f in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        # speech/eos token from HF config
        self.config.eos = self.config.hf_config.eos_token_id

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # prefer bfloat16 when available for memory savings on supported hardware
        torch_dtype = getattr(torch, "bfloat16", None)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype)
        # move to device if loaded on CPU
        try:
            self.model.to(self.device)
        except Exception:
            # some HF device_map configs already place the model; ignore failures
            pass

        self.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompt: List[int], sampling_param: SamplingParams, past_key_values: Optional[object] = None) -> dict:
        stopping_criteria = StoppingCriteriaList([EosTokenCriteria(eos_token_id=self.config.hf_config.eos_token_id)])

        if getattr(sampling_param, "use_ras", False):
            sample_hf_engine_handler = partial(
                _ras_sample_hf_engine,
                use_ras=sampling_param.use_ras,
                win_size=sampling_param.win_size,
                tau_r=sampling_param.tau_r,
            )
        else:
            sample_hf_engine_handler = None

        rep_pen_processor = RepetitionPenaltyLogitsProcessor(
            penalty=sampling_param.repetition_penalty, prompt_ignore_length=len(prompt)
        )

        with torch.no_grad():
            input_ids = torch.tensor([prompt], dtype=torch.int64).to(self.device)
            input_len = input_ids.shape[1]
            generated = self.model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=sampling_param.top_k,
                top_p=sampling_param.top_p,
                min_new_tokens=sampling_param.min_tokens,
                max_new_tokens=sampling_param.max_tokens,
                temperature=sampling_param.temperature,
                stopping_criteria=stopping_criteria,
                past_key_values=past_key_values,
                custom_generate=sample_hf_engine_handler,
                use_cache=True,
                logits_processor=[rep_pen_processor],
            )
            # strip the prompt portion and return ids
            generated_ids = generated[:, input_len:].cpu().numpy().tolist()[0]

        return {"text": self.tokenizer.decode(generated_ids, skip_special_tokens=True), "token_ids": generated_ids}


class VLLMEngine:
    """vLLM-based engine wrapper. Raises ImportError if vLLM isn't available."""

    def __init__(self, model: str, **kwargs) -> None:
        if not SUPPORT_VLLM:
            raise ImportError("vLLM is not installed or supported in this environment")

        config_fields = {f.name for f in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.config.eos = self.config.hf_config.eos_token_id
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # vLLM configuration and instance
        os.environ.setdefault("VLLM_USE_V1", "0")
        self.model = LLM(model=model, enforce_eager=True, dtype="bfloat16", max_model_len=8192)
        self.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompt: List[int], sampling_param: SamplingParams, past_key_values: Optional[object] = None) -> dict:
        # ensure the vLLM stop ids are set
        sampling_param.stop_token_ids = [self.config.hf_config.eos_token_id]

        # build vLLM prompt and sampling params then generate
        vllm_prompt = TokensPrompt(prompt_token_ids=prompt)
        vllm_params = VllmSamplingParams(**asdict(sampling_param))

        # vLLM returns a sequence-like object; extract token ids from the first result
        result = self.model.generate(vllm_prompt, vllm_params, use_tqdm=False)
        generated_ids = result[0].outputs[0].token_ids

        return {"text": self.tokenizer.decode(list(generated_ids), skip_special_tokens=True), "token_ids": list(generated_ids)}
