# -*- coding: utf-8 -*-
"""
Utility functions for SoulX-Podcast WebUI.
"""

import re
from typing import Optional, List


# =============================================================================
# Type Coercion Functions
# =============================================================================

def coerce_gradio_file_to_path(file_obj) -> Optional[str]:
    """
    兼容 gr.File 的不同返回形态：
    - str 路径
    - dict（含 name/path）
    - 对象（含 name 属性）
    """
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("name") or file_obj.get("path")
    return getattr(file_obj, "name", None)


def coerce_audio_value_to_path(audio_val) -> Optional[str]:
    """
    gr.Audio(type="filepath") 通常返回 str 路径；这里做额外兼容。
    """
    if audio_val is None:
        return None
    if isinstance(audio_val, str):
        return audio_val
    if isinstance(audio_val, dict):
        return audio_val.get("name") or audio_val.get("path")
    return getattr(audio_val, "name", None)


# =============================================================================
# Text Validation Functions
# =============================================================================

def check_monologue_text(text: str, prefix: str = None) -> bool:
    """Check if monologue text is valid."""
    text = text.strip()
    # Check speaker tags
    if prefix is not None and (not text.startswith(prefix)):
        return False
    # Remove prefix
    if prefix is not None:
        text = text.removeprefix(prefix)
    text = text.strip()
    # If empty?
    if len(text) == 0:
        return False
    return True


def check_dialect_prompt_text(text: str, prefix: str = None) -> bool:
    """Check if dialect prompt text is valid."""
    text = text.strip()
    # Check Dialect Prompt prefix tags
    if prefix is not None and (not text.startswith(prefix)):
        return False
    text = text.strip()
    # If empty?
    if len(text) == 0:
        return False
    return True


def check_dialogue_text(text_list: List[str], max_speakers: int = None) -> bool:
    """Check if dialogue text list is valid."""
    if len(text_list) == 0:
        return False
    for text in text_list:
        # 检查是否匹配 [S1] 到 [S{max_speakers}] 格式
        pattern = r'^\[S([1-9]|[1-9][0-9]+)\].*'
        match = re.match(pattern, text.strip(), flags=re.DOTALL)
        if not match:
            return False
        spk_num = int(match.group(1))
        if spk_num < 1:
            return False
        if max_speakers is not None and spk_num > max_speakers:
            return False
    return True

