# -*- coding: utf-8 -*-
"""
Internationalization (i18n) support for SoulX-Podcast WebUI.
"""

from typing import Literal

# =============================================================================
# Global Language State
# =============================================================================

global_lang: Literal["zh", "en"] = "zh"


def set_language(lang: Literal["zh", "en"]):
    """Set the global language."""
    global global_lang
    global_lang = lang


def get_language() -> Literal["zh", "en"]:
    """Get the current global language."""
    global global_lang
    return global_lang


# =============================================================================
# Internationalization Dictionary
# =============================================================================

_i18n_key2lang_dict = dict(
    # Speaker1 Prompt
    spk1_prompt_audio_label=dict(
        en="Speaker 1 Prompt Audio",
        zh="说话人 1 参考语音",
    ),
    spk1_prompt_text_label=dict(
        en="Speaker 1 Prompt Text",
        zh="说话人 1 参考文本",
    ),
    spk1_prompt_text_placeholder=dict(
        en="text of speaker 1 Prompt audio.",
        zh="说话人 1 参考文本",
    ),
    spk1_dialect_prompt_text_label=dict(
        en="Speaker 1 Dialect Prompt Text",
        zh="说话人 1 方言提示文本",
    ),
    spk1_dialect_prompt_text_placeholder=dict(
        en="Dialect prompt text with prefix: <|Sichuan|>/<|Yue|>/<|Henan|> ",
        zh="带前缀方言提示词思维链文本，前缀如下：<|Sichuan|>/<|Yue|>/<|Henan|>，如：<|Sichuan|>走嘛，切吃那家新开的麻辣烫，听别个说味道硬是霸道得很，好吃到不摆了，去晚了还得排队！",
    ),
    # Speaker2 Prompt
    spk2_prompt_audio_label=dict(
        en="Speaker 2 Prompt Audio",
        zh="说话人 2 参考语音",
    ),
    spk2_prompt_text_label=dict(
        en="Speaker 2 Prompt Text",
        zh="说话人 2 参考文本",
    ),
    spk2_prompt_text_placeholder=dict(
        en="text of speaker 2 prompt audio.",
        zh="说话人 2 参考文本",
    ),
    spk2_dialect_prompt_text_label=dict(
        en="Speaker 2 Dialect Prompt Text",
        zh="说话人 2 方言提示文本",
    ),
    spk2_dialect_prompt_text_placeholder=dict(
        en="Dialect prompt text with prefix: <|Sichuan|>/<|Yue|>/<|Henan|> ",
        zh="带前缀方言提示词思维链文本，前缀如下：<|Sichuan|>/<|Yue|>/<|Henan|>，如：<|Sichuan|>走嘛，切吃那家新开的麻辣烫，听别个说味道硬是霸道得很，好吃到不摆了，去晚了还得排队！",
    ),
    # Speaker3 Prompt
    spk3_prompt_audio_label=dict(
        en="Speaker 3 Prompt Audio",
        zh="说话人 3 参考语音",
    ),
    spk3_prompt_text_label=dict(
        en="Speaker 3 Prompt Text",
        zh="说话人 3 参考文本",
    ),
    spk3_prompt_text_placeholder=dict(
        en="text of speaker 3 Prompt audio.",
        zh="说话人 3 参考文本",
    ),
    spk3_dialect_prompt_text_label=dict(
        en="Speaker 3 Dialect Prompt Text",
        zh="说话人 3 方言提示文本",
    ),
    spk3_dialect_prompt_text_placeholder=dict(
        en="Dialect prompt text with prefix: <|Sichuan|>/<|Yue|>/<|Henan|> ",
        zh="带前缀方言提示词思维链文本，前缀如下：<|Sichuan|>/<|Yue|>/<|Henan|>，如：<|Sichuan|>走嘛，切吃那家新开的麻辣烫，听别个说味道硬是霸道得很，好吃到不摆了，去晚了还得排队！",
    ),
    # Speaker4 Prompt
    spk4_prompt_audio_label=dict(
        en="Speaker 4 Prompt Audio",
        zh="说话人 4 参考语音",
    ),
    spk4_prompt_text_label=dict(
        en="Speaker 4 Prompt Text",
        zh="说话人 4 参考文本",
    ),
    spk4_prompt_text_placeholder=dict(
        en="text of speaker 4 Prompt audio.",
        zh="说话人 4 参考文本",
    ),
    spk4_dialect_prompt_text_label=dict(
        en="Speaker 4 Dialect Prompt Text",
        zh="说话人 4 方言提示文本",
    ),
    spk4_dialect_prompt_text_placeholder=dict(
        en="Dialect prompt text with prefix: <|Sichuan|>/<|Yue|>/<|Henan|> ",
        zh="带前缀方言提示词思维链文本，前缀如下：<|Sichuan|>/<|Yue|>/<|Henan|>，如：<|Sichuan|>走嘛，切吃那家新开的麻辣烫，听别个说味道硬是霸道得很，好吃到不摆了，去晚了还得排队！",
    ),
    # Dialogue input textbox
    dialogue_text_input_label=dict(
        en="Dialogue Text Input",
        zh="合成文本输入",
    ),
    dialogue_text_input_placeholder=dict(
        en="[S1]text[S2]text[S3]text... (Use [S1], [S2], [S3], etc. to specify speakers)",
        zh="[S1]文本[S2]文本[S3]文本... (使用 [S1], [S2], [S3] 等指定说话人)",
    ),
    # Generate button
    generate_btn_label=dict(
        en="Generate Audio",
        zh="合成",
    ),
    # Generated audio
    generated_audio_label=dict(
        en="Generated Dialogue Audio",
        zh="合成的对话音频",
    ),
    # Warining1: invalid text for prompt
    warn_invalid_spk1_prompt_text=dict(
        en='Invalid speaker 1 prompt text, should not be empty and strictly follow: "xxx"',
        zh='说话人 1 参考文本不合规，不能为空，格式："xxx"',
    ),
    warn_invalid_spk2_prompt_text=dict(
        en='Invalid speaker 2 prompt text, should strictly follow: "[S2]xxx"',
        zh='说话人 2 参考文本不合规，格式："[S2]xxx"',
    ),
    warn_invalid_dialogue_text=dict(
        en='Invalid dialogue input text, should strictly follow: "[S1]xxx[S2]xxx..."',
        zh='对话文本输入不合规，格式："[S1]xxx[S2]xxx..."',
    ),
    # Warining3: incomplete prompt info
    warn_incomplete_prompt=dict(
        en="Please provide prompt audio and text for all speakers used in the dialogue",
        zh="请为对话中使用的所有说话人提供参考语音与参考文本",
    ),
    # Speaker manage controls
    add_speaker_btn_label=dict(
        en="Add 1 Speaker",
        zh="添加1个说话人",
    ),
    quick_add_num_label=dict(
        en="Quick Add Count",
        zh="快速添加数量",
    ),
    quick_add_btn_label=dict(
        en="Quick Add",
        zh="快速添加",
    ),
    select_all_btn_label=dict(
        en="Select All",
        zh="全选",
    ),
    select_none_btn_label=dict(
        en="Select None",
        zh="全不选",
    ),
    batch_delete_btn_label=dict(
        en="Delete Selected",
        zh="批量删除选中",
    ),
    # Separated audio files info
    separated_files_info_label=dict(
        en="Separated Audio Files Info",
        zh="分离音频文件信息",
    ),
    separated_files_info_placeholder=dict(
        en="Separated speaker audio files will be saved in outputs/separated_speakers/ directory",
        zh="分离的说话者音频文件将保存在 outputs/separated_speakers/ 目录下",
    ),
    # Download files
    download_all_files_label=dict(
        en="Download All Audio Files (ZIP)",
        zh="下载所有音频文件 (ZIP)",
    ),
    # File info messages
    files_saved_to=dict(
        en="Audio files saved to:",
        zh="音频文件已保存到:",
    ),
    files_generated_count=dict(
        en="Files generated this time (total: {count}):",
        zh="本次生成的文件 (共 {count} 个):",
    ),
    complete_dialogue_audio=dict(
        en="Complete Dialogue Audio",
        zh="整体对话音频",
    ),
    speaker_label=dict(
        en="Speaker {num}",
        zh="说话者 {num}",
    ),
    complete_audio_label=dict(
        en="(Complete Audio)",
        zh="(完整音频)",
    ),
    zip_file_created=dict(
        en="Zip file created: {filename}",
        zh="压缩包已创建: {filename}",
    ),
    download_hint=dict(
        en="(You can download all files below)",
        zh="(可在下方下载所有文件)",
    ),
    no_files_saved=dict(
        en="(No files saved, may be disabled or error occurred)",
        zh="（未保存文件，可能已禁用或出现错误）",
    ),
    # Different speaker pause
    diff_spk_pause_label=dict(
        en="Different-speaker pause (ms)",
        zh="不同说话者间停顿(ms)",
    ),
    task_pause_label=dict(
        en="Task interval pause (ms)",
        zh="任务间停顿(ms)",
    ),
    # Log messages (for console)
    log_saved_complete_dialogue=dict(
        en="Saved complete dialogue audio",
        zh="已保存整体对话音频",
    ),
    log_saved_speaker_complete=dict(
        en="Saved speaker {num} complete audio",
        zh="已保存说话者 {num} 完整音频",
    ),
    log_saved_speaker_part=dict(
        en="Saved speaker {num} part {part}",
        zh="已保存说话者 {num} 片段 {part}",
    ),
    log_all_files_saved=dict(
        en="All audio files saved to: {dir}",
        zh="所有音频文件已保存到: {dir}",
    ),
    log_total_files_saved=dict(
        en="Total {count} files saved",
        zh="共保存 {count} 个文件",
    ),
    log_file_added_to_zip=dict(
        en="Added file to zip: {filename}",
        zh="已添加文件到压缩包: {filename}",
    ),
    log_zip_created=dict(
        en="Zip file created: {filename}",
        zh="压缩包已创建: {filename}",
    ),
    log_error_saving_files=dict(
        en="Error saving audio files: {error}",
        zh="保存音频文件时出错: {error}",
    ),
    log_error_creating_zip=dict(
        en="Error creating zip file: {error}",
        zh="创建压缩包时出错: {error}",
    ),
)


# =============================================================================
# i18n Functions
# =============================================================================

def i18n(key: str) -> str:
    """Get internationalized text for a given key."""
    global global_lang
    if key in _i18n_key2lang_dict:
        return _i18n_key2lang_dict[key][global_lang]
    return key


def get_i18n_dict() -> dict:
    """Get the full i18n dictionary (for checking key existence)."""
    return _i18n_key2lang_dict


def get_select_speaker_label(idx: int) -> str:
    """返回带语言的 选择说话人/Select Speaker 标签。"""
    global global_lang
    if global_lang == "en":
        return f"Select Speaker {idx}"
    return f"选择说话人 {idx}"


def get_speaker_display_label(idx: int, remark: str = "") -> str:
    """
    返回用于 Tab/复选框显示的标签。
    若有备注，则显示为 S{idx}:备注，否则回退到语言化的“选择说话人 {idx}”。
    """
    remark_clean = (remark or "").strip()
    if remark_clean:
        return f"S{idx}:{remark_clean}"
    return get_select_speaker_label(idx)

