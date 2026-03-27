from __future__ import annotations

from contextlib import nullcontext

import torch


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_llm_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def get_amp_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    if device.type == "mps":
        return torch.amp.autocast("mps", dtype=torch.float16)
    return nullcontext()
