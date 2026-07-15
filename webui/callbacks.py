# -*- coding: utf-8 -*-
"""
UI callback functions for SoulX-Podcast WebUI.
"""

import re
import os
import time
from datetime import datetime
from typing import List

import numpy as np
import gradio as gr
import soundfile as sf

from .constants import MAX_SPEAKERS, MAX_TEXT_INPUTS
from .i18n import (
    i18n,
    get_i18n_dict,
    get_speaker_display_label,
    get_language,
    set_language,
)
from .synthesis import dialogue_synthesis_function
from .file_manager import create_all_zip, write_json_file
from .config_manager import build_current_config_dict


# =============================================================================
# Speaker Management Callbacks
# =============================================================================

def update_speakers_visibility(num_speakers: int, remarks=None):
    """更新说话人列的可见性和标签"""
    remark_list = list(remarks) if remarks else []
    updates = []
    for i in range(MAX_SPEAKERS):
        visible = (i < num_speakers)
        remark_val = remark_list[i] if i < len(remark_list) else ""
        label = get_speaker_display_label(i + 1, remark_val)
        if visible:
            updates.append(gr.update(visible=True, label=label, value=False))
        else:
            updates.append(gr.update(visible=False, value=False))
    return updates


def add_speaker(current_num: int, *remarks):
    """添加一个说话人"""
    remark_list = list(remarks) if remarks else []
    new_num = min(current_num + 1, MAX_SPEAKERS)
    checkbox_updates = update_speakers_visibility(new_num, remark_list)
    column_updates = []
    for i in range(MAX_SPEAKERS):
        remark_val = remark_list[i] if i < len(remark_list) else ""
        column_updates.append(
            gr.update(
                visible=(i < new_num),
                label=get_speaker_display_label(i + 1, remark_val)
            )
        )
    return new_num, *checkbox_updates, *column_updates


def quick_add_speakers(current_num: int, add_count, *remarks):
    """快速添加指定数量的说话人"""
    remark_list = list(remarks) if remarks else []
    add_count = int(add_count) if add_count else 1
    add_count = max(1, min(add_count, MAX_SPEAKERS - current_num))
    new_num = min(current_num + add_count, MAX_SPEAKERS)
    checkbox_updates = update_speakers_visibility(new_num, remark_list)
    column_updates = []
    for i in range(MAX_SPEAKERS):
        remark_val = remark_list[i] if i < len(remark_list) else ""
        column_updates.append(
            gr.update(
                visible=(i < new_num),
                label=get_speaker_display_label(i + 1, remark_val)
            )
        )
    return new_num, *checkbox_updates, *column_updates


def batch_delete_speakers(current_num: int, *all_values):
    """批量删除选中的说话人，并重新排列剩余说话人及其数据"""
    # all_values格式: (checkbox1, audio1, text1, dialect1, remark1, checkbox2, ...)
    speaker_data = []
    for i in range(MAX_SPEAKERS):
        base = i * 5
        checkbox_val = all_values[base] if base < len(all_values) else False
        audio_val = all_values[base + 1] if base + 1 < len(all_values) else None
        text_val = all_values[base + 2] if base + 2 < len(all_values) else ""
        dialect_val = all_values[base + 3] if base + 3 < len(all_values) else ""
        remark_val = all_values[base + 4] if base + 4 < len(all_values) else ""
        speaker_data.append(
            dict(
                checkbox=checkbox_val,
                audio=audio_val,
                text=text_val,
                dialect=dialect_val,
                remark=remark_val,
            )
        )
    
    selected_indices = {
        i for i, spk in enumerate(speaker_data) if spk["checkbox"] and i < current_num
    }
    
    def _build_updates(target_num: int, kept: list):
        updates = []
        for i in range(MAX_SPEAKERS):
            if i < target_num and i < len(kept):
                spk = kept[i]
                label = get_speaker_display_label(i + 1, spk["remark"])
                updates.extend(
                    [
                        gr.update(visible=True, label=label, value=False),
                        gr.update(value=spk["audio"]),
                        gr.update(value=spk["text"]),
                        gr.update(value=spk["dialect"]),
                        gr.update(value=spk["remark"]),
                    ]
                )
            else:
                updates.extend(
                    [
                        gr.update(visible=False, value=False),
                        gr.update(value=None),
                        gr.update(value=""),
                        gr.update(value=""),
                        gr.update(value=""),
                    ]
                )
        tab_updates = [
            gr.update(
                visible=(i < target_num),
                label=get_speaker_display_label(
                    i + 1, kept[i]["remark"] if i < len(kept) else ""
                ),
            )
            for i in range(MAX_SPEAKERS)
        ]
        return updates, tab_updates
    
    if not selected_indices:
        gr.Warning("请至少选择一个说话人进行删除")
        kept_list = [speaker_data[i] for i in range(current_num)]
        updates, tab_updates = _build_updates(current_num, kept_list)
        return current_num, *updates, *tab_updates
    
    remaining_count = current_num - len(selected_indices)
    if remaining_count < 1:
        gr.Warning("至少需要保留1个说话人")
        kept_list = [speaker_data[i] for i in range(current_num)]
        updates, tab_updates = _build_updates(current_num, kept_list)
        return current_num, *updates, *tab_updates
    
    kept_indices = [i for i in range(current_num) if i not in selected_indices]
    kept_list = [speaker_data[i] for i in kept_indices]
    updates, tab_updates = _build_updates(remaining_count, kept_list)
    return remaining_count, *updates, *tab_updates


def select_all_checkboxes(current_num: int):
    """全选所有可见的复选框"""
    updates = []
    for i in range(MAX_SPEAKERS):
        if i < current_num:
            updates.append(gr.update(value=True))
        else:
            updates.append(gr.update())
    return updates


def select_none_checkboxes(current_num: int):
    """取消全选所有复选框"""
    updates = []
    for i in range(MAX_SPEAKERS):
        updates.append(gr.update(value=False))
    return updates


def update_single_speaker_label(remark: str, idx: int):
    """根据备注更新单个说话人的复选框与 Tab 标签"""
    label = get_speaker_display_label(idx, remark)
    return gr.update(label=label), gr.update(label=label)


def refresh_all_speaker_labels_after_load(num_speakers: int, *remarks):
    """
    配置加载后，显式刷新所有说话人的复选框和Tab标签
    这个函数用于解决配置加载时标签不立即更新的问题
    """
    from datetime import datetime
    current_time = datetime.now().strftime('%H-%M-%S')
    print(f"[{current_time}] 刷新所有说话人标签...")
    
    remark_list = list(remarks) if remarks else []
    num = int(num_speakers) if num_speakers else 1
    num = max(1, min(num, MAX_SPEAKERS))
    
    checkbox_updates = []
    tab_updates = []
    
    for i in range(MAX_SPEAKERS):
        remark_val = remark_list[i] if i < len(remark_list) else ""
        label = get_speaker_display_label(i + 1, remark_val)
        
        if i < num:
            checkbox_updates.append(gr.update(label=label, visible=True))
            tab_updates.append(gr.update(label=label, visible=True))
        else:
            checkbox_updates.append(gr.update(label=label, visible=False))
            tab_updates.append(gr.update(label=label, visible=False))
    
    print(f"[{current_time}] 已刷新 {num} 个说话人的标签")
    return (*checkbox_updates, *tab_updates)


def update_speaker_accordion_label(num_speakers: int, *remarks):
    """
    更新说话人设置 Accordion 的标题，显示所有说话人的标签信息
    """
    remark_list = list(remarks) if remarks else []
    num = int(num_speakers) if num_speakers else 1
    num = max(1, min(num, MAX_SPEAKERS))
    
    # 构建说话人标签列表
    speaker_labels = []
    for i in range(num):
        remark_val = remark_list[i] if i < len(remark_list) else ""
        label = get_speaker_display_label(i + 1, remark_val)
        speaker_labels.append(label)
    
    # 生成标题
    if speaker_labels:
        labels_str = ", ".join(speaker_labels)
        title = f"👥 说话人设置 / Speakers ({labels_str})"
    else:
        title = "👥 说话人设置 / Speakers"
    
    return gr.update(label=title)


def _build_speaker_labels(num_speakers: int, remarks=None):
    """生成当前可见说话人的标签列表"""
    remark_list = list(remarks) if remarks else []
    labels = []
    for i in range(max(1, min(int(num_speakers) if num_speakers else 1, MAX_SPEAKERS))):
        remark_val = remark_list[i] if i < len(remark_list) else ""
        labels.append(get_speaker_display_label(i + 1, remark_val))
    return labels


def update_speaker_selection_choices(num_speakers: int, *remarks):
    """更新快捷勾选组件的选项"""
    labels = _build_speaker_labels(num_speakers, remarks)
    return gr.update(choices=labels, value=[])


def selection_group_to_checkboxes(selected_labels, num_speakers: int, *remarks):
    """将快捷勾选结果同步到各说话人复选框"""
    labels = _build_speaker_labels(num_speakers, remarks)
    selected_set = set(selected_labels or [])
    updates = []
    for i in range(MAX_SPEAKERS):
        if i < len(labels):
            updates.append(gr.update(value=(labels[i] in selected_set), visible=True))
        else:
            updates.append(gr.update(value=False, visible=False))
    return updates


def select_all_selection_group(num_speakers: int, *remarks):
    """同步全选到快捷勾选组件"""
    labels = _build_speaker_labels(num_speakers, remarks)
    return gr.update(value=labels)


def select_none_selection_group():
    """同步全不选到快捷勾选组件"""
    return gr.update(value=[])


# =============================================================================
# Text Input Management
# =============================================================================

def update_text_inputs_visibility(num_inputs):
    """更新文本输入框的可见性"""
    num = int(num_inputs) if num_inputs else 1
    num = max(1, min(num, MAX_TEXT_INPUTS))
    updates = []
    audio_updates = []
    download_updates = []
    for i in range(MAX_TEXT_INPUTS):
        is_visible = (i < num)
        updates.append(gr.update(
            visible=is_visible,
            label=f"{i18n('dialogue_text_input_label')} {i+1}"
        ))
        # 预览组件应该和文本输入框保持相同的可见性
        # 这样当有音频生成时，预览组件才能正确显示
        audio_updates.append(gr.update(visible=is_visible))
        download_updates.append(gr.update(visible=False))
    return num, *updates, *audio_updates, *download_updates


# =============================================================================
# Synthesis Processing
# =============================================================================

def process_single_synthesis(
    target_text: str,
    num_speakers: int,
    seed: int,
    diff_spk_pause_ms: int,
    speaker_args: List,
    task_number: int,
    base_output_dir: str,
    timestamp: str,
):
    """
    处理单个合成任务
    task_number: 任务编号（从1开始）
    base_output_dir: 基础输出目录（时间戳文件夹）
    timestamp: 统一的时间戳
    Returns: (audio_result, saved_files, zip_file_path, output_dir, task_time_seconds)
    """
    task_start_time = time.time()
    current_time = datetime.now().strftime('%H-%M-%S')
    
    speaker_configs = []
    # 支持两种格式：3个一组（audio, text, dialect）或4个一组（audio, text, dialect, remark）
    # 根据参数数量自动判断格式
    step = 4 if len(speaker_args) >= num_speakers * 4 else 3
    for i in range(0, min(num_speakers * step, len(speaker_args)), step):
        if i + 2 < len(speaker_args):
            audio = speaker_args[i] if speaker_args[i] is not None else None
            text = speaker_args[i+1] if speaker_args[i+1] is not None else ""
            dialect = speaker_args[i+2] if speaker_args[i+2] is not None else ""
            speaker_configs.append((text, audio, dialect))
    
    task_subdir = f"{task_number:03d}"
    output_dir = os.path.join(base_output_dir, task_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[{current_time}] 开始处理任务 {task_number}")
    
    try:
        result = dialogue_synthesis_function(
            target_text,
            speaker_configs,
            seed,
            int(diff_spk_pause_ms) if diff_spk_pause_ms is not None else 0,
            output_dir=output_dir,
            save_separated=True,
            timestamp=timestamp
        )
        
        task_end_time = time.time()
        task_time = task_end_time - task_start_time
        current_time_end = datetime.now().strftime('%H-%M-%S')
        
        if result is None:
            # dialogue_synthesis_function 返回 None 表示失败
            print(f"[{current_time_end}] 任务 {task_number} 处理失败，耗时: {task_time:.2f} 秒")
            return None, [], None, output_dir, task_time
        
        audio_result, saved_files = result
        print(f"[{current_time_end}] 任务 {task_number} 处理完成，耗时: {task_time:.2f} 秒")
        return audio_result, saved_files, None, output_dir, task_time
    except Exception as e:
        task_end_time = time.time()
        task_time = task_end_time - task_start_time
        current_time_end = datetime.now().strftime('%H-%M-%S')
        error_msg = f"process_single_synthesis 执行失败: {str(e)}"
        print(f"[{current_time_end}] [ERROR] {error_msg}")
        print(f"[{current_time_end}] 任务 {task_number} 处理失败，耗时: {task_time:.2f} 秒")
        import traceback
        traceback.print_exc()
        return None, [], None, output_dir, task_time


def write_log_to_file(log_content: str, log_file_path: str):
    """将日志内容写入文件"""
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_content + '\n')
    except Exception as e:
        print(f"[WARNING] 写入日志文件失败: {str(e)}")


def collect_and_synthesize_queue(
    num_text_inputs,
    num_speakers,
    seed,
    diff_spk_pause_ms,
    task_pause_ms,
    language_idx,
    *all_text_and_speaker_args
):
    """
    处理队列中的所有任务（生成器版本，每完成一个任务就更新预览）
    all_text_and_speaker_args格式: (text1, ..., textN, audio1, text1, dialect1, remark1, audio2, text2, dialect2, remark2, ...)
    language_idx: 语言索引 (0=中文, 1=English 或 "中文"/"English")
    task_pause_ms: 任务间的停顿时间（毫秒）
    """
    global_lang = get_language()
    num_text = int(num_text_inputs) if num_text_inputs else 1
    num_speaker = int(num_speakers)
    task_pause_seconds = (int(task_pause_ms) if task_pause_ms is not None else 500) / 1000.0
    
    text_inputs = list(all_text_and_speaker_args[:MAX_TEXT_INPUTS])
    speaker_args = list(all_text_and_speaker_args[MAX_TEXT_INPUTS:])
    
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(text_inputs[:num_text]):
        if text and text.strip():
            valid_texts.append(text)
            valid_indices.append(i)
    
    if not valid_texts:
        empty_audio_updates = [gr.update(visible=False) for _ in range(MAX_TEXT_INPUTS)]
        empty_download_updates = [gr.update(visible=False) for _ in range(MAX_TEXT_INPUTS)]
        yield (
            None,
            "所有输入框均为空，请至少填写一个文本输入",
            gr.update(visible=False),
            gr.update(interactive=True),  # Left generate button
            gr.update(interactive=True),  # Right generate button
            *empty_audio_updates,
            *empty_download_updates,
        )
        return
    
    total_start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = os.path.join(os.getcwd(), "outputs", "separated_speakers", timestamp)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 创建日志文件
    log_file_path = os.path.join(base_output_dir, "synthesis.log")
    current_time = datetime.now().strftime('%H-%M-%S')
    initial_log = f"[{current_time}] 开始处理 {len(valid_texts)} 个任务\n"
    initial_log += f"[{current_time}] 输出目录: {os.path.abspath(base_output_dir)}\n"
    initial_log += f"[{current_time}] 任务间停顿时间: {task_pause_ms if task_pause_ms is not None else 500} ms\n"
    write_log_to_file(initial_log, log_file_path)
    
    # 只显示总体进度，不显示每个任务的进度
    # 使用 track_tqdm=False 避免在每个文本框下显示进度条
    progress_bar = gr.Progress(track_tqdm=False)
    all_info_messages = []
    task_audio_results = {}
    all_complete_audio_files = []
    all_generated_files = []
    task_times = []  # 记录每个任务的耗时
    
    # 初始化所有预览为不可见
    audio_preview_updates = [gr.update(visible=False) for _ in range(MAX_TEXT_INPUTS)]
    download_updates = [gr.update(visible=False) for _ in range(MAX_TEXT_INPUTS)]
    
    for task_idx, (text_idx, target_text) in enumerate(zip(valid_indices, valid_texts)):
        # 不显示每个任务的进度，避免在每个文本框下显示进度条
        # 只在开始时显示一次总体进度
        if task_idx == 0 and len(valid_texts) > 1:
            progress_bar(0, desc=f"开始处理 {len(valid_texts)} 个任务")
        
        task_start_time = time.time()
        try:
            task_number = task_idx + 1
            audio_result, saved_files, zip_file_path, output_dir, task_time = process_single_synthesis(
                target_text, num_speaker, seed, diff_spk_pause_ms, speaker_args,
                task_number, base_output_dir, timestamp
            )
            
            task_times.append(task_time)
            current_time = datetime.now().strftime('%H-%M-%S')
            task_log_msg = f"[{current_time}] 任务 {task_idx + 1} (输入框 {text_idx + 1}) 处理完成，耗时: {task_time:.2f} 秒"
            write_log_to_file(task_log_msg, log_file_path)
            
            # 检查处理是否成功
            if audio_result is None or not saved_files:
                error_msg = f"任务 {task_idx + 1} (输入框 {text_idx + 1}) 处理失败，未生成音频文件，耗时: {task_time:.2f} 秒"
                all_info_messages.append(error_msg)
                print(f"[WARNING] {error_msg}")
                write_log_to_file(f"[{current_time}] [WARNING] {error_msg}", log_file_path)
                continue
            
            task_audio_results[text_idx] = audio_result
            all_generated_files.extend(saved_files)
            
            print(f"[INFO] 任务 {task_idx + 1} (输入框 {text_idx + 1}) 完成，音频已生成")
            
            complete_files = [f for f in saved_files if "complete_dialogue" in os.path.basename(f)]
            if complete_files:
                all_complete_audio_files.extend(complete_files)
            
            task_subdir_name = f"{task_number:03d}"
            info_message = f"═══════════════════════════════════\n"
            info_message += f"任务 {task_idx + 1}/{len(valid_texts)} (输入框 {text_idx + 1})\n"
            info_message += f"═══════════════════════════════════\n"
            info_message += f"⏱️ 处理时间: {task_time:.2f} 秒\n"
            info_message += f"{i18n('files_saved_to')}\n"
            info_message += f"基础文件夹: {os.path.abspath(base_output_dir)}\n"
            info_message += f"任务子文件夹: {task_subdir_name}/\n"
            info_message += f"完整路径: {os.path.abspath(output_dir)}\n\n"
            
            if saved_files:
                info_message += f"{i18n('files_generated_count').format(count=len(saved_files))}\n\n"
                
                complete_files = [f for f in saved_files if "complete_dialogue" in os.path.basename(f)]
                if complete_files:
                    info_message += f"📁 {i18n('complete_dialogue_audio')}:\n"
                    for f in complete_files:
                        info_message += f"  • {os.path.basename(f)}\n"
                    info_message += "\n"
                
                speaker_groups = {}
                for f in saved_files:
                    basename = os.path.basename(f)
                    if "speaker" in basename and "complete_dialogue" not in basename:
                        match = re.search(r'speaker(\d+)', basename)
                        if match:
                            spk_num = match.group(1)
                            if spk_num not in speaker_groups:
                                speaker_groups[spk_num] = []
                            speaker_groups[spk_num].append(basename)
                
                for spk_num in sorted(speaker_groups.keys(), key=int):
                    files = sorted(speaker_groups[spk_num])
                    complete_audio = [f for f in files if "_complete_" in f]
                    parts = [f for f in files if "_part" in f]
                    
                    info_message += f"🎤 {i18n('speaker_label').format(num=spk_num)}:\n"
                    if complete_audio:
                        for filename in complete_audio:
                            info_message += f"  • {filename} {i18n('complete_audio_label')}\n"
                    if parts:
                        for filename in sorted(parts):
                            info_message += f"  • {filename}\n"
                    info_message += "\n"
            else:
                info_message += f"{i18n('no_files_saved')}\n"
            
            all_info_messages.append(info_message)
            
            # 每完成一个任务，立即更新该任务的预览
            current_info_message = f"📂 所有任务文件保存在统一的时间戳文件夹中:\n"
            current_info_message += f"   {os.path.abspath(base_output_dir)}\n"
            current_info_message += f"   每个任务的文件保存在对应的编号子文件夹中 (001/, 002/, 003/, ...)\n"
            current_info_message += f"   分段语音保存在各任务子文件夹的 separated/ 子文件夹中\n"
            current_info_message += "\n"
            current_info_message += "═══════════════════════════════════\n\n"
            current_info_message += "\n\n".join(all_info_messages)
            current_info_message += f"\n\n⏳ 进行中: 已完成 {task_idx + 1}/{len(valid_texts)} 个任务"
            
            # 更新当前任务的预览
            # 预览组件应该和文本输入框保持相同的可见性
            # 这样当有音频生成时，预览组件才能正确显示
            current_audio_preview_updates = []
            current_download_updates = []
            for i in range(MAX_TEXT_INPUTS):
                # 检查这个输入框是否在有效输入框中（即文本输入框是否可见）
                is_text_input_visible = i in valid_indices or i < num_text
                
                if i in task_audio_results:
                    # 有音频结果，显示预览
                    if global_lang == "zh":
                        audio_label = f"任务 {i+1} 音频预览"
                    else:
                        audio_label = f"Task {i+1} Audio Preview"
                    
                    print(f"[INFO] 更新预览组件 {i+1}: 显示音频预览")
                    
                    current_audio_preview_updates.append(gr.update(
                        visible=True,
                        value=task_audio_results[i],
                        label=audio_label
                    ))
                    current_download_updates.append(gr.update(visible=False))
                elif is_text_input_visible:
                    # 文本输入框可见但还没有音频，保持预览组件可见（显示为空）
                    if global_lang == "zh":
                        audio_label = f"任务 {i+1} 音频预览"
                    else:
                        audio_label = f"Task {i+1} Audio Preview"
                    
                    current_audio_preview_updates.append(gr.update(
                        visible=True,
                        value=None,
                        label=audio_label
                    ))
                    current_download_updates.append(gr.update(visible=False))
                else:
                    # 文本输入框不可见，预览组件也不可见
                    current_audio_preview_updates.append(gr.update(visible=False))
                    current_download_updates.append(gr.update(visible=False))
            
            # 计算当前合并音频（如果有多个任务已完成）
            current_preview_audio_value = None
            if len(all_complete_audio_files) > 0 and task_idx == 0:
                # 第一个任务完成时，使用第一个任务的音频作为预览
                current_preview_audio_value = audio_result
            elif len(all_complete_audio_files) > 1:
                # 多个任务完成时，尝试合并已完成的音频
                try:
                    temp_merged_path = os.path.join(base_output_dir, "temp_merged.wav")
                    merged_audio_data = None
                    sample_rate = 24000
                    
                    for idx, audio_file in enumerate(all_complete_audio_files):
                        if os.path.exists(audio_file):
                            audio_data, sr = sf.read(audio_file)
                            if sample_rate != sr:
                                print(f"[WARNING] 采样率不一致: {audio_file} 为 {sr}Hz，期望 {sample_rate}Hz")
                            
                            if len(audio_data.shape) > 1:
                                audio_data = np.mean(audio_data, axis=1)
                            
                            if merged_audio_data is None:
                                merged_audio_data = audio_data
                            else:
                                # 使用可配置的任务间停顿时间
                                pause_samples = int(task_pause_seconds * sample_rate)
                                silence = np.zeros(pause_samples)
                                merged_audio_data = np.concatenate([merged_audio_data, silence, audio_data])
                    
                    if merged_audio_data is not None:
                        sf.write(temp_merged_path, merged_audio_data, sample_rate)
                        current_preview_audio_value = (sample_rate, merged_audio_data)
                except Exception as e:
                    print(f"[WARNING] 临时合并音频失败: {str(e)}")
                    if task_audio_results:
                        current_preview_audio_value = list(task_audio_results.values())[-1]
            elif task_audio_results:
                current_preview_audio_value = list(task_audio_results.values())[-1]
            
            # 每完成一个任务就 yield 一次更新
            yield (
                current_preview_audio_value,
                current_info_message,
                gr.update(visible=False),  # 下载文件在最后才生成
                gr.update(interactive=False),  # Left generate button (处理中禁用)
                gr.update(interactive=False),  # Right generate button (处理中禁用)
                *current_audio_preview_updates,
                *current_download_updates,
            )
            
            # 如果不是最后一个任务，添加任务间停顿
            if task_idx < len(valid_texts) - 1 and task_pause_seconds > 0:
                current_time = datetime.now().strftime('%H-%M-%S')
                pause_log = f"[{current_time}] 任务间停顿 {task_pause_seconds:.2f} 秒..."
                write_log_to_file(pause_log, log_file_path)
                time.sleep(task_pause_seconds)
            
        except Exception as e:
            task_end_time = time.time()
            task_time = task_end_time - task_start_time
            current_time = datetime.now().strftime('%H-%M-%S')
            error_msg = f"任务 {task_idx + 1} 处理失败: {str(e)}\n"
            error_msg += f"耗时: {task_time:.2f} 秒"
            all_info_messages.append(error_msg)
            write_log_to_file(f"[{current_time}] [ERROR] {error_msg}", log_file_path)
            task_times.append(task_time)  # 即使失败也记录时间
            import traceback
            traceback.print_exc()
    
    # 合并所有任务的完整对话音频
    merged_audio_path = None
    if all_complete_audio_files and len(all_complete_audio_files) > 0:
        try:
            current_time = datetime.now().strftime('%H-%M-%S')
            merge_start_time = time.time()
            write_log_to_file(f"[{current_time}] 开始合并所有任务音频...", log_file_path)
            
            merged_audio_path = os.path.join(base_output_dir, "all.wav")
            merged_audio_data = None
            sample_rate = 24000
            
            for idx, audio_file in enumerate(all_complete_audio_files):
                if os.path.exists(audio_file):
                    audio_data, sr = sf.read(audio_file)
                    if sample_rate != sr:
                        print(f"[WARNING] 采样率不一致: {audio_file} 为 {sr}Hz，期望 {sample_rate}Hz")
                    
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    
                    if merged_audio_data is None:
                        merged_audio_data = audio_data
                    else:
                        # 使用可配置的任务间停顿时间
                        pause_samples = int(task_pause_seconds * sample_rate)
                        silence = np.zeros(pause_samples)
                        merged_audio_data = np.concatenate([merged_audio_data, silence, audio_data])
            
            if merged_audio_data is not None:
                sf.write(merged_audio_path, merged_audio_data, sample_rate)
                merge_time = time.time() - merge_start_time
                current_time = datetime.now().strftime('%H-%M-%S')
                print(f"[INFO] 已合并所有任务音频到: {merged_audio_path}")
                write_log_to_file(f"[{current_time}] 音频合并完成，耗时: {merge_time:.2f} 秒", log_file_path)
                all_generated_files.append(merged_audio_path)
        except Exception as e:
            current_time = datetime.now().strftime('%H-%M-%S')
            print(f"[ERROR] 合并音频文件时出错: {str(e)}")
            write_log_to_file(f"[{current_time}] [ERROR] 合并音频文件时出错: {str(e)}", log_file_path)
            import traceback
            traceback.print_exc()
    
    # 更新总体进度为完成
    if len(valid_texts) > 1:
        progress_bar(1.0, desc=f"已完成所有 {len(valid_texts)} 个任务")
    
    # 计算总耗时
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    current_time = datetime.now().strftime('%H-%M-%S')
    
    # 记录总耗时和各任务耗时到日志
    total_log = f"\n[{current_time}] {'='*50}\n"
    total_log += f"[{current_time}] 所有任务处理完成\n"
    total_log += f"[{current_time}] 总任务数: {len(valid_texts)}\n"
    if task_times:
        total_log += f"[{current_time}] 各任务耗时: "
        for i, t in enumerate(task_times, 1):
            total_log += f"任务{i}({t:.2f}s) "
        total_log += "\n"
        avg_time = sum(task_times) / len(task_times)
        total_log += f"[{current_time}] 平均任务耗时: {avg_time:.2f} 秒\n"
    total_log += f"[{current_time}] 总耗时: {total_time:.2f} 秒\n"
    total_log += f"[{current_time}] {'='*50}\n"
    write_log_to_file(total_log, log_file_path)
    
    # 创建 all.zip
    all_zip_path = None
    if all_generated_files:
        all_zip_path = create_all_zip(base_output_dir, all_generated_files)
    
    # 构建最终信息
    final_info_message = f"📂 所有任务文件保存在统一的时间戳文件夹中:\n"
    final_info_message += f"   {os.path.abspath(base_output_dir)}\n"
    final_info_message += f"   每个任务的文件保存在对应的编号子文件夹中 (001/, 002/, 003/, ...)\n"
    final_info_message += f"   分段语音保存在各任务子文件夹的 separated/ 子文件夹中\n"
    if merged_audio_path and os.path.exists(merged_audio_path):
        final_info_message += f"   📁 合并音频文件: {os.path.basename(merged_audio_path)}\n"
    if all_zip_path and os.path.exists(all_zip_path):
        final_info_message += f"   📦 所有文件压缩包: {os.path.basename(all_zip_path)}\n"
    final_info_message += f"   📝 日志文件: synthesis.log\n"
    final_info_message += f"   ⚙️ 配置文件: config.json\n"
    final_info_message += "\n"
    final_info_message += "═══════════════════════════════════\n\n"
    final_info_message += "\n\n".join(all_info_messages)
    final_info_message += f"\n\n{'='*50}\n"
    final_info_message += f"⏱️ 总处理时间: {total_time:.2f} 秒\n"
    if task_times:
        final_info_message += f"⏱️ 各任务耗时: "
        for i, t in enumerate(task_times, 1):
            final_info_message += f"任务{i}({t:.2f}s) "
        final_info_message += "\n"
        avg_time = sum(task_times) / len(task_times)
        final_info_message += f"⏱️ 平均任务耗时: {avg_time:.2f} 秒\n"
    final_info_message += f"✅ 已完成所有任务 ({len(valid_texts)}/{len(valid_texts)})\n"
    
    # 生成最终更新
    final_audio_preview_updates = []
    final_download_updates = []
    
    preview_audio_value = None
    if merged_audio_path and os.path.exists(merged_audio_path):
        try:
            audio_data, sample_rate = sf.read(merged_audio_path)
            preview_audio_value = (sample_rate, audio_data)
        except Exception as e:
            print(f"[WARNING] 读取 all.wav 文件失败: {str(e)}")
            if task_audio_results:
                preview_audio_value = list(task_audio_results.values())[-1]
    elif task_audio_results:
        preview_audio_value = list(task_audio_results.values())[-1]
    
    for i in range(MAX_TEXT_INPUTS):
        # 检查这个输入框是否在有效输入框中（即文本输入框是否可见）
        is_text_input_visible = i in valid_indices or i < num_text
        
        if i in task_audio_results:
            # 有音频结果，显示预览
            if global_lang == "zh":
                audio_label = f"任务 {i+1} 音频预览"
            else:
                audio_label = f"Task {i+1} Audio Preview"
            
            final_audio_preview_updates.append(gr.update(
                visible=True,
                value=task_audio_results[i],
                label=audio_label
            ))
            final_download_updates.append(gr.update(visible=False))
        elif is_text_input_visible:
            # 文本输入框可见但还没有音频，保持预览组件可见（显示为空）
            if global_lang == "zh":
                audio_label = f"任务 {i+1} 音频预览"
            else:
                audio_label = f"Task {i+1} Audio Preview"
            
            final_audio_preview_updates.append(gr.update(
                visible=True,
                value=None,
                label=audio_label
            ))
            final_download_updates.append(gr.update(visible=False))
        else:
            # 文本输入框不可见，预览组件也不可见
            final_audio_preview_updates.append(gr.update(visible=False))
            final_download_updates.append(gr.update(visible=False))
    
    # 导出配置到任务目录
    try:
        current_time = datetime.now().strftime('%H-%M-%S')
        print(f"[{current_time}] 开始导出配置到任务目录...")
        
        # 构建配置字典
        # speaker_args 格式：每4个一组 (audio, text, dialect, remark)
        # 但需要确保长度符合 MAX_SPEAKERS * 4 的要求
        speaker_values = list(speaker_args)
        
        # 确保 speaker_values 长度符合要求（MAX_SPEAKERS * 4）
        # 如果不足，补齐空值
        expected_length = MAX_SPEAKERS * 4
        while len(speaker_values) < expected_length:
            # 根据位置判断应该填充什么类型的值
            idx = len(speaker_values)
            if idx % 4 == 0:
                # audio 位置，填充 None
                speaker_values.append(None)
            else:
                # text, dialect, remark 位置，填充空字符串
                speaker_values.append("")
        
        config_dict = build_current_config_dict(
            language_idx=language_idx,
            seed=seed,
            diff_spk_pause_ms=diff_spk_pause_ms,
            task_pause_ms=task_pause_ms,
            num_speakers=num_speaker,
            num_text_inputs=num_text,
            text_inputs=text_inputs,
            speaker_values=speaker_values,
        )
        
        # 保存配置到任务目录
        config_file_path = os.path.join(base_output_dir, "config.json")
        write_json_file(config_file_path, config_dict)
        current_time = datetime.now().strftime('%H-%M-%S')
        print(f"[{current_time}] ✅ 配置已导出到: {config_file_path}")
        write_log_to_file(f"[{current_time}] ✅ 配置已导出到: {config_file_path}", log_file_path)
    except Exception as e:
        current_time = datetime.now().strftime('%H-%M-%S')
        error_msg = f"导出配置失败: {str(e)}"
        print(f"[{current_time}] [ERROR] {error_msg}")
        write_log_to_file(f"[{current_time}] [ERROR] {error_msg}", log_file_path)
        import traceback
        traceback.print_exc()
    
    download_file_update = None
    if all_zip_path and os.path.exists(all_zip_path):
        download_label = f"{i18n('download_all_files_label')} - all.zip"
        download_file_update = gr.update(visible=True, value=all_zip_path, label=download_label)
    else:
        download_file_update = gr.update(visible=False, value=None)
    
    # 最后一次 yield，返回最终结果
    yield (
        preview_audio_value,
        final_info_message,
        download_file_update,
        gr.update(interactive=True),  # Left generate button
        gr.update(interactive=True),  # Right generate button
        *final_audio_preview_updates,
        *final_download_updates,
    )


# =============================================================================
# Language Switch Callback
# =============================================================================

def change_component_language(lang, *remarks):
    """Change language for all components."""
    if isinstance(lang, str):
        set_language("zh" if lang == "中文" else "en")
    else:
        try:
            set_language(["zh", "en"][int(lang)])
        except Exception:
            set_language("zh")
    global_lang = get_language()
    i18n_dict = get_i18n_dict()
    
    checkbox_updates = []
    input_updates = []
    
    remark_list = list(remarks) if remarks else []
    for i in range(MAX_SPEAKERS):
        remark_val = remark_list[i] if i < len(remark_list) else ""
        checkbox_updates.append(gr.update(label=get_speaker_display_label(i + 1, remark_val)))
    
    for i in range(MAX_SPEAKERS):
        input_updates.extend([
            gr.update(label=i18n(f"spk{i+1}_prompt_audio_label") if f"spk{i+1}_prompt_audio_label" in i18n_dict else f"说话人 {i+1} 参考语音"),
            gr.update(
                label=i18n(f"spk{i+1}_prompt_text_label") if f"spk{i+1}_prompt_text_label" in i18n_dict else f"说话人 {i+1} 参考文本",
            ),
            gr.update(
                label=i18n(f"spk{i+1}_dialect_prompt_text_label") if f"spk{i+1}_dialect_prompt_text_label" in i18n_dict else f"说话人 {i+1} 方言提示文本",
            ),
        ])
    
    updates = checkbox_updates + input_updates
    
    for i in range(MAX_TEXT_INPUTS):
        updates.append(gr.update(
            label=f"{i18n('dialogue_text_input_label')} {i+1}",
        ))
    
    for i in range(MAX_TEXT_INPUTS):
        if global_lang == "zh":
            updates.append(gr.update(label=f"任务 {i+1} 音频预览"))
            updates.append(gr.update(label=f"任务 {i+1} 下载"))
        else:
            updates.append(gr.update(label=f"Task {i+1} Audio Preview"))
            updates.append(gr.update(label=f"Task {i+1} Download"))
    
    updates.extend([
        gr.update(value=i18n("generate_btn_label")),  # Left generate button
        gr.update(value=i18n("generate_btn_label")),  # Right generate button
        gr.update(label=i18n("generated_audio_label")),
        gr.update(value=f"➕ {i18n('add_speaker_btn_label')}"),
        gr.update(label=i18n('quick_add_num_label')),
        gr.update(value=f"🚀 {i18n('quick_add_btn_label')}"),
        gr.update(value=f"☑️ {i18n('select_all_btn_label')}"),
        gr.update(value=f"☐ {i18n('select_none_btn_label')}"),
        gr.update(value=f"🗑️ {i18n('batch_delete_btn_label')}"),
        gr.update(
            label=i18n("separated_files_info_label"),
            placeholder=i18n("separated_files_info_placeholder"),
        ),
        gr.update(label=i18n("download_all_files_label")),
        gr.update(label=i18n("diff_spk_pause_label")),
        gr.update(label=i18n("task_pause_label")),
    ])
    return updates

