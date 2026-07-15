# -*- coding: utf-8 -*-
"""
UI component creation functions for SoulX-Podcast WebUI.
"""

import gradio as gr

from .constants import DIALECT_PROMPT_DATA
from .i18n import get_speaker_display_label


# =============================================================================
# Speaker Components
# =============================================================================

def create_speaker_group(spk_num: int):
    """创建一个说话人组件组"""
    with gr.Group(visible=True) as group:
        # 添加复选框用于选择删除
        checkbox = gr.Checkbox(
            label=get_speaker_display_label(spk_num),
            value=False,
            scale=0,
        )
        remark = gr.Textbox(
            label="备注名",
            placeholder="例如：佩奇",
            lines=1,
        )
        prompt_audio = gr.Audio(
            label=f"说话人 {spk_num} 参考语音",
            type="filepath",
            editable=False,
            interactive=True,
        )
        prompt_text = gr.Textbox(
            label=f"说话人 {spk_num} 参考文本",
            placeholder=f"说话人 {spk_num} 参考文本",
            lines=3,
        )
        dialect_prompt_text = gr.Textbox(
            label=f"说话人 {spk_num} 方言提示文本",
            placeholder="带前缀方言提示词思维链文本，前缀如下：<|Sichuan|>/<|Yue|>/<|Henan|>",
            value="",
            lines=3,
        )
    return group, checkbox, remark, prompt_audio, prompt_text, dialect_prompt_text


# =============================================================================
# Dialect Selection Functions
# =============================================================================

def update_example_choices(dialect_key: str):
    """Update example choices based on selected dialect."""
    if dialect_key == "(无)":
        choices = ["(请先选择方言)"]
        return gr.update(choices=choices, value="(无)"), gr.update(choices=choices, value="(无)")
    
    choices = list(DIALECT_PROMPT_DATA.get(dialect_key, {}).keys())
    return gr.update(choices=choices, value="(无)"), gr.update(choices=choices, value="(无)")


def update_prompt_text(dialect_key: str, example_key: str):
    """Update prompt text based on dialect and example selection."""
    if dialect_key == "(无)" or example_key in ["(无)", "(请先选择方言)"]:
        return gr.update(value="")
    
    full_text = DIALECT_PROMPT_DATA.get(dialect_key, {}).get(example_key, "")
    return gr.update(value=full_text)

