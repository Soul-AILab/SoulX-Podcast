# -*- coding: utf-8 -*-
"""
Core synthesis logic for SoulX-Podcast WebUI.
"""

import re
import os
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Optional

import torch
import numpy as np
import gradio as gr
import s3tokenizer
import soundfile as sf

from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.config import Config, SamplingParams
from soulxpodcast.utils.dataloader import PodcastInferHandler

from .i18n import i18n
from .utils import check_dialogue_text


# =============================================================================
# Global Model State
# =============================================================================

model: SoulXPodcast = None
dataset: PodcastInferHandler = None


def get_model() -> SoulXPodcast:
    """Get the global model instance."""
    global model
    return model


def get_dataset() -> PodcastInferHandler:
    """Get the global dataset instance."""
    global dataset
    return dataset


def initiate_model(config: Config, enable_tn: bool = False):
    """Initialize the global model and dataset."""
    global model
    if model is None:
        model = SoulXPodcast(config)

    global dataset
    if dataset is None:
        dataset = PodcastInferHandler(model.llm.tokenizer, None, config)


# =============================================================================
# Data Processing
# =============================================================================

def process_single(
    target_text_list: List[str],
    prompt_wav_list: List[str],
    prompt_text_list: List[str],
    use_dialect_prompt: bool,
    dialect_prompt_text: List[str],
) -> dict:
    """Process a single synthesis request."""
    spks, texts = [], []
    for target_text in target_text_list:
        pattern = r'(\[S([1-9]|[1-9][0-9]+)\])(.+)'
        match = re.match(pattern, target_text, flags=re.DOTALL)
        if not match:
            print(f"process_single: target_text: {target_text}, not match")
            continue
        spk_num = int(match.group(2))
        text = match.group(3).strip()
        # print(f"process_single: target_text: \nstart{target_text}\nend, spk_num: {spk_num},\n text: \nstart{text}\nend")
        spk = spk_num - 1  # S1->0, S2->1, etc.
        spks.append(spk)
        texts.append(text)
    
    # 检查是否成功解析出文本和说话人
    if not texts or not spks:
        error_msg = f"process_single: 未能从 target_text_list 中解析出有效的文本或说话人。spks={spks}, texts={texts}"
        print(error_msg)
        raise ValueError(error_msg)
    
    global dataset
    dataitem = {
        "key": "001",
        "prompt_text": prompt_text_list,
        "prompt_wav": prompt_wav_list,
        "text": texts,
        "spk": spks,
    }
    if use_dialect_prompt:
        dataitem.update({
            "dialect_prompt_text": dialect_prompt_text
        })
    dataset.update_datasource([dataitem])

    # assert one data only;
    data = dataset[0]
    if data is None:
        error_msg = "process_single: dataset[0] 返回 None，数据处理失败。请检查音频文件路径和格式是否正确。"
        print(error_msg)
        raise ValueError(error_msg)
    
    prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(data["log_mel"])
    spk_emb_for_flow = torch.tensor(data["spk_emb"])
    prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(
        data["mel"], batch_first=True, padding_value=0
    )
    prompt_mels_lens_for_flow = torch.tensor(data['mel_len'])
    text_tokens_for_llm = data["text_tokens"]
    prompt_text_tokens_for_llm = data["prompt_text_tokens"]
    spk_ids = data["spks_list"]
    sampling_params = SamplingParams(use_ras=True, win_size=25, tau_r=0.2)
    infos = [data["info"]]
    
    processed_data = {
        "prompt_mels_for_llm": prompt_mels_for_llm,
        "prompt_mels_lens_for_llm": prompt_mels_lens_for_llm,
        "prompt_text_tokens_for_llm": prompt_text_tokens_for_llm,
        "text_tokens_for_llm": text_tokens_for_llm,
        "prompt_mels_for_flow_ori": prompt_mels_for_flow,
        "prompt_mels_lens_for_flow": prompt_mels_lens_for_flow,
        "spk_emb_for_flow": spk_emb_for_flow,
        "sampling_params": sampling_params,
        "spk_ids": spk_ids,
        "infos": infos,
        "use_dialect_prompt": use_dialect_prompt,
    }
    if use_dialect_prompt:
        processed_data.update({
            "dialect_prompt_text_tokens_for_llm": data["dialect_prompt_text_tokens"],
            "dialect_prefix": data["dialect_prefix"],
        })
    return processed_data


# =============================================================================
# Core Synthesis Function
# =============================================================================

def dialogue_synthesis_function(
    target_text: str,
    speaker_configs_list: List[Tuple[str, str, str]],
    seed: int = 1988,
    diff_spk_pause_ms: int = 0,
    output_dir: Optional[str] = None,
    save_separated: bool = True,
    timestamp: Optional[str] = None,
):
    """
    合成对话音频
    speaker_configs_list: 说话人配置列表，每个元素为 (prompt_text, prompt_audio, dialect_prompt_text)
    output_dir: 输出目录，用于保存分离的说话者音频文件
    save_separated: 是否保存分离的说话者音频文件
    timestamp: 时间戳（如果不提供则自动生成）
    """
    import random
    
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Check prompt info
    # 首先按行分割文本，记录每个片段属于哪一行
    lines = target_text.split('\n')
    
    # 匹配 [S1]... 到下一个 [Sx] 或文本结尾
    pattern = r'\[S([1-9]|[1-9][0-9]+)\](.*?)(?=\[S([1-9]|[1-9][0-9]+)\]|$)'
    
    # 重新组合完整匹配：说话人标签 + 内容
    target_text_list: List[str] = []
    spk_seq: List[int] = []
    pause_after_ms_list: List[int] = []
    line_indices: List[int] = []
    pause_token_pattern = re.compile(r'<\|pause:(\d+)\|>')
    
    # 按行处理文本，记录每个片段属于哪一行
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        matches = list(re.finditer(pattern, line, re.DOTALL))
        for match in matches:
            spk_num_str = match.group(1)
            content = match.group(2)
            try:
                spk_num_int = int(spk_num_str)
            except Exception:
                spk_num_int = -1
            # 按停顿标记拆分内容
            parts = re.split(r'(<\|pause:\d+\|>)', content)
            last_idx_with_text = None
            for p in parts:
                if p is None or p == '':
                    continue
                pause_m = pause_token_pattern.fullmatch(p.strip())
                if pause_m is not None:
                    if last_idx_with_text is not None:
                        try:
                            pause_ms = int(pause_m.group(1))
                        except Exception:
                            pause_ms = 0
                        pause_after_ms_list[last_idx_with_text] = max(0, pause_ms)
                    continue
                text_part = p.strip()
                if len(text_part) == 0:
                    continue
                full_text = f"[S{spk_num_str}]{text_part}"
                target_text_list.append(full_text)
                spk_seq.append(spk_num_int)
                pause_after_ms_list.append(0)
                line_indices.append(line_idx)
                last_idx_with_text = len(target_text_list) - 1
    
    # 找出对话中使用的最大说话人编号
    max_spk_used = 0
    for text in target_text_list:
        match = re.match(r'\[S([1-9]|[1-9][0-9]+)\]', text)
        if match:
            spk_num = int(match.group(1))
            max_spk_used = max(max_spk_used, spk_num)
    
    if max_spk_used == 0:
        gr.Warning(message="对话文本中未找到有效的说话人标签（[S1], [S2]等）")
        return None
    
    num_speakers = len(speaker_configs_list)
    if max_spk_used > num_speakers:
        gr.Warning(message=f"对话中使用了[S{max_spk_used}]，但只提供了{num_speakers}个说话人配置")
        return None
    
    if not check_dialogue_text(target_text_list, max_speakers=num_speakers):
        print(f"dialogue_synthesis_function: target_text_list: {target_text_list}, not match")
        gr.Warning(message=i18n("warn_invalid_dialogue_text"))
        return None

    # 检查所有使用的说话人是否都有配置
    for i in range(max_spk_used):
        if i >= len(speaker_configs_list):
            gr.Warning(message=f"说话人 {i+1} 缺少配置")
            return None
        config = speaker_configs_list[i]
        if not config[1] or not config[0]:
            gr.Warning(message=f"说话人 {i+1} 缺少参考语音或参考文本")
            return None

    # Go synthesis
    # 使用 track_tqdm=False 避免影响外部预览框的显示
    progress_bar = gr.Progress(track_tqdm=False)
    prompt_wav_list = [config[1] for config in speaker_configs_list[:max_spk_used]]
    prompt_text_list = [config[0] for config in speaker_configs_list[:max_spk_used]]
    use_dialect_prompt = any(config[2].strip() != "" for config in speaker_configs_list[:max_spk_used])
    dialect_prompt_text_list = [config[2] for config in speaker_configs_list[:max_spk_used]]
    
    try:
        data = process_single(
            target_text_list,
            prompt_wav_list,
            prompt_text_list,
            use_dialect_prompt,
            dialect_prompt_text_list,
        )
    except Exception as e:
        error_msg = f"处理数据时出错: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        gr.Warning(message=error_msg)
        return None, []
    
    if data is None:
        error_msg = "process_single 返回 None，数据处理失败"
        print(f"[ERROR] {error_msg}")
        gr.Warning(message=error_msg)
        return None, []
    
    try:
        global model
        results_dict = model.forward_longform(**data)
    except Exception as e:
        error_msg = f"模型推理时出错: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        gr.Warning(message=error_msg)
        return None, []
    
    target_audio = None
    sample_rate = 24000
    num_segments = len(results_dict['generated_wavs'])
    saved_files = []
    
    # 验证片段数量是否匹配
    if num_segments != len(spk_seq):
        print(f"[WARNING] 音频片段数量 ({num_segments}) 与说话者序列长度 ({len(spk_seq)}) 不匹配")
    
    # 按顺序记录生成的音频片段
    ordered_segment_infos: List[dict] = []
    speaker_part_counter: dict[int, int] = defaultdict(int)
    
    for i in range(num_segments):
        seg = results_dict['generated_wavs'][i]
        
        if i < len(spk_seq):
            current_speaker = spk_seq[i]
            if current_speaker > 0:
                speaker_part_counter[current_speaker] += 1
                part_idx = speaker_part_counter[current_speaker]
                seg_copy = seg.clone().detach()
                current_line_idx = line_indices[i] if i < len(line_indices) else -1
                ordered_segment_infos.append({
                    "turn_index": i,
                    "speaker": current_speaker,
                    "part_idx": part_idx,
                    "audio": seg_copy,
                    "line_idx": current_line_idx,
                })
        
        # 合并到整体音频
        if target_audio is None:
            target_audio = seg
        else:
            prefer_ms = 0
            if i > 0 and (i - 1) < len(pause_after_ms_list):
                prefer_ms = int(pause_after_ms_list[i - 1])
            if prefer_ms <= 0:
                insert_silence = False
                if i > 0 and (i - 1) < len(spk_seq) and i < len(spk_seq):
                    prev_spk = spk_seq[i - 1]
                    curr_spk = spk_seq[i]
                    insert_silence = (prev_spk != curr_spk)
                if insert_silence and diff_spk_pause_ms and diff_spk_pause_ms > 0:
                    prefer_ms = int(diff_spk_pause_ms)
            if prefer_ms and prefer_ms > 0:
                silence_len = int((prefer_ms / 1000.0) * sample_rate)
                if silence_len > 0:
                    silence = torch.zeros((1, silence_len), dtype=seg.dtype, device=seg.device)
                    target_audio = torch.concat([target_audio, silence], dim=1)
            target_audio = torch.concat([target_audio, seg], dim=1)
    
    # 保存音频文件
    if save_separated and output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            separated_dir = os.path.join(output_dir, "separated")
            os.makedirs(separated_dir, exist_ok=True)
            
            sentences_dir = os.path.join(output_dir, "sentences")
            os.makedirs(sentences_dir, exist_ok=True)
            
            file_counter = 1
            
            # 保存整体音频文件
            if target_audio is not None:
                complete_audio_filename = os.path.join(output_dir, "complete_dialogue.wav")
                sf.write(complete_audio_filename, target_audio.cpu().squeeze(0).numpy(), sample_rate)
                saved_files.append(complete_audio_filename)
                print(f"[INFO] {i18n('log_saved_complete_dialogue')}: {complete_audio_filename}")
            
            # 按对话顺序保存分离的说话者音频片段
            if ordered_segment_infos:
                for seg_info in ordered_segment_infos:
                    seg_audio_np = seg_info["audio"].cpu().squeeze(0).numpy()
                    part_filename = os.path.join(
                        separated_dir,
                        f"{file_counter:03d}_speaker{seg_info['speaker']}_part{seg_info['part_idx']}.wav"
                    )
                    sf.write(part_filename, seg_audio_np, sample_rate)
                    saved_files.append(part_filename)
                    print(f"[INFO] {i18n('log_saved_speaker_part').format(num=seg_info['speaker'], part=seg_info['part_idx'])}: {part_filename}")
                    file_counter += 1
            
            # 按行合并音频片段并保存
            if ordered_segment_infos:
                line_to_segments: dict[int, List[dict]] = defaultdict(list)
                for idx, seg_info in enumerate(ordered_segment_infos):
                    line_idx = seg_info.get("line_idx", -1)
                    if line_idx >= 0:
                        seg_info_with_pause = seg_info.copy()
                        turn_index = seg_info.get("turn_index", idx)
                        if turn_index < len(pause_after_ms_list):
                            seg_info_with_pause["pause_after_ms"] = pause_after_ms_list[turn_index]
                        else:
                            seg_info_with_pause["pause_after_ms"] = 0
                        line_to_segments[line_idx].append(seg_info_with_pause)
                
                valid_line_indices = sorted(line_to_segments.keys())
                
                sentence_counter = 1
                for line_idx in valid_line_indices:
                    segments = line_to_segments[line_idx]
                    if not segments:
                        continue
                    
                    line_audio = None
                    for seg_idx, seg_info in enumerate(segments):
                        seg_audio = seg_info["audio"]
                        
                        if line_audio is None:
                            line_audio = seg_audio
                        else:
                            if seg_idx > 0:
                                prev_pause_ms = segments[seg_idx - 1].get("pause_after_ms", 0)
                                if prev_pause_ms > 0:
                                    silence_len = int((prev_pause_ms / 1000.0) * sample_rate)
                                    if silence_len > 0:
                                        silence = torch.zeros((1, silence_len), dtype=seg_audio.dtype, device=seg_audio.device)
                                        line_audio = torch.concat([line_audio, silence], dim=1)
                            line_audio = torch.concat([line_audio, seg_audio], dim=1)
                    
                    if line_audio is not None:
                        sentence_filename = os.path.join(sentences_dir, f"sentence{sentence_counter}.wav")
                        sf.write(sentence_filename, line_audio.cpu().squeeze(0).numpy(), sample_rate)
                        saved_files.append(sentence_filename)
                        print(f"[INFO] 已保存按行合并音频 sentence{sentence_counter}: {sentence_filename}")
                        sentence_counter += 1
            
            if saved_files:
                print(f"[INFO] {i18n('log_all_files_saved').format(dir=output_dir)}")
                print(f"[INFO] {i18n('log_total_files_saved').format(count=len(saved_files))}")
        except Exception as e:
            print(f"[ERROR] {i18n('log_error_saving_files').format(error=str(e))}")
            import traceback
            traceback.print_exc()
    
    return (sample_rate, target_audio.cpu().squeeze(0).numpy()), saved_files

