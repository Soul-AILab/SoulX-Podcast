import re
import json
import torch
import argparse
from tqdm import tqdm
from datetime import datetime

import s3tokenizer

from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.utils.dataloader import PodcastInferHandler
from soulxpodcast.utils.commons import set_all_random_seed
from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams


def initiate_model(seed, model_path, llm_engine, fp16_flow):
    set_all_random_seed(seed)
    
    hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
        initial_values={"fp16_flow": fp16_flow},
        json_file=f"{model_path}/soulxpodcast_config.json"
    )
    if llm_engine == "vllm":
        import importlib.util
        if not importlib.util.find_spec("vllm"):
            llm_engine = "hf"
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [WARNING]: No install VLLM, switch to hf engine.")

    config = Config(model=model_path, enforce_eager=True, llm_engine=llm_engine, hf_config=hf_config)
    model = SoulXPodcast(config)

    dataset = PodcastInferHandler(model.llm.tokenizer, None, config)
    
    return model, dataset


def process_single_input(dataset, target_text_list, prompt_wav_list, prompt_text_list, use_dialect_prompt, dialect_prompt_text_list):
    spks, texts = [], []
    for target_text in target_text_list:
        pattern = r'(\[S[1-9]\])(.+)'
        match = re.match(pattern, target_text)
        text, spk = match.group(2), int(match.group(1)[2])-1
        spks.append(spk)
        texts.append(text)
    
    dataitem = {"key": "001", "prompt_text": prompt_text_list, "prompt_wav": prompt_wav_list, 
             "text": texts, "spk": spks, }
    if use_dialect_prompt:
        dataitem.update({
            "dialect_prompt_text": dialect_prompt_text_list
        })
    dataset.update_datasource(
        [
           dataitem 
        ]
    )        

    # assert one data only;
    data = dataset[0]
    prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(data["log_mel"])  # [B, num_mels=128, T]
    spk_emb_for_flow = torch.tensor(data["spk_emb"])
    prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(data["mel"], batch_first=True, padding_value=0)  # [B, T', num_mels=80]
    prompt_mels_lens_for_flow = torch.tensor(data['mel_len'])
    text_tokens_for_llm = data["text_tokens"]
    prompt_text_tokens_for_llm = data["prompt_text_tokens"]
    spk_ids = data["spks_list"]
    sampling_params = SamplingParams(use_ras=True,win_size=25,tau_r=0.2)
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


def check_models(model_path, inputs):
    if inputs['use_dialect_prompt']:
        assert 'dialect' in model_path, "Dialect prompt is used, you should use a dialect model."
    
    return True