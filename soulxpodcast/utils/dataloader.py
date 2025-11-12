import os
import json
from tqdm import tqdm
from datetime import datetime

import onnxruntime
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torchaudio.compliance.kaldi as kaldi

import s3tokenizer

from soulxpodcast.utils.text import normalize_text
from soulxpodcast.utils.audio import mel_spectrogram, audio_volume_normalize
from soulxpodcast.utils.prompt_cache import PROMPT_FEATURE_CACHE
from soulxpodcast.utils.timing import Timer, TIMING_COLLECTOR
from soulxpodcast.config import Config, SamplingParams


SPK_DICT = ["<|SPEAKER_0|>", "<|SPEAKER_1|>", "<|SPEAKER_2|>", "<|SPEAKER_3|>",]
TEXT_START, TEXT_END, AUDIO_START = "<|text_start|>", "<|text_end|>", "<|semantic_token_start|>" 
TASK_PODCAST = "<|task_podcast|>"


class PodcastDataset(Dataset):

    def __init__(self, text_tokenizer, data_list, model_config: Config):
        self.datas = []
        self.model_config = model_config

        """Example data_list:
        ```
        {"key": "uttid_1", "prompt_text": ["prompt_text1", "prompt_text2"], "dialect_prompt_text": ["dialect_prompt_text1", "dialect_prompt_text2"],
            "text": ["text1", "text2], "spk": [0, 1], "prompt_wav": ["/mnt/data/audio/00000000.wav", "/mnt/data/audio/00000001.wav"], "wav": "/mnt/data/audio_synthetic/uttid_1.wav"}
        ```
        Note:
            - `key` is the key of this sample.
            - `prompt_text` is the text used for prompt.
            - `dialect_prompt_text` is the reshot text used for prompt.
            - `text` is the text used for generating real audio.
            - `spk` is the target speaker id to synthesize, corresponds to the prompt order. Default SPEAKER_0.
            - `prompt_wav` is the audio used for prompt.
            - `wav` is the path to the generated audio to be saved (we highly recommend to pre-define the save path before running the script).
        """
        missing = 0
        with open(data_list, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
            iterator = tqdm(lines, desc='Loading data')
            for line in iterator:
                data = json.loads(line.strip())
                valid = True
                for k in ['key', 'prompt_text', 'text', 'prompt_wav']:
                    if k not in data:
                        valid = False
                        break
                    if data[k] is None:
                        valid = False
                        break
                valid = True
                for url in data["prompt_wav"]:
                    if not os.path.exists(url):
                        valid = False
                        break
                if valid:
                    self.datas.append(data)
                else:
                    missing += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        tqdm.write(f'[{timestamp}] - [INFO] - Loaded {total_lines} lines, found {missing} missing lines, total valid lines == {len(self.datas)}.')

        self.text_tokenizer = text_tokenizer

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(f"{self.model_config.model}/campplus.onnx", sess_options=option,
                                                      providers=["CPUExecutionProvider"])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        try:
            prompt_text_ids_list, dialect_prompt_text_ids_list, spk_emb_list, mel_list, mel_len_list, log_mel_list = (
                [], [], [], [], [], []
            )
            # Prepare prompt information
            use_dialect_prompt = "dialect_prompt_text" in data
            dialect_prefix_list = []
            dialect_prefix_list.append(self.text_tokenizer.encode(f"{TASK_PODCAST}"))
            for spk_idx, (prompt_text, prompt_wav) in enumerate(zip(data["prompt_text"], data["prompt_wav"])):
                # Try cache first (avoids repeated audio loading / feature extraction)
                # Try to load a persisted entry from disk if memory cache miss
                cached = PROMPT_FEATURE_CACHE.get(prompt_wav)
                if cached is None:
                    try:
                        cached = PROMPT_FEATURE_CACHE.load_entry_if_exists(prompt_wav)
                    except Exception:
                        cached = None

                if cached is not None:
                    # debug log for cache hit
                    try:
                        from datetime import datetime
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                        tqdm.write(f'[{timestamp}] - [cache hit] using precomputed prompt features: {prompt_wav}')
                    except Exception:
                        pass
                    # record a quick cache hit metric
                    TIMING_COLLECTOR.record('cache_hit', 0.0)

                    # cached keys: audio_16k, log_mel, mel, mel_len, spk_emb
                    log_mel = cached["log_mel"]
                    spk_emb = cached["spk_emb"]
                    mel = cached["mel"]
                    mel_len = cached["mel_len"]
                else:
                    # 1. feature for s3tokenizer (log mel at 16k)
                    with Timer('feature_load'):
                        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)
                        audio = audio_volume_normalize(audio)
                        # [T]
                        log_mel = s3tokenizer.log_mel_spectrogram(audio)  # [num_mels, T]

                    # 2. feature for speaker embedding
                    with Timer('spk_embed'):
                        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
                        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
                        spk_emb = self.spk_model.run(
                            None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
                        )[0].flatten().tolist()

                    # 3. feature for flow (mel at 24k)
                    with Timer('flow_mel'):
                        audio_flow, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
                        audio_flow = audio_flow[0]
                        audio_flow = audio_volume_normalize(audio_flow).unsqueeze(0)
                        if sample_rate != 24000:
                            audio_flow = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio_flow)
                        mel = mel_spectrogram(audio_flow).transpose(1, 2).squeeze(0)  # [T, num_mels]
                        if mel.shape[0] % 2 != 0:
                            mel = mel[:-1]
                        mel_len = mel.shape[0]

                    # store into cache for future reuse
                    try:
                        PROMPT_FEATURE_CACHE.add(prompt_wav, {
                            "audio_16k": audio.cpu() if hasattr(audio, "cpu") else audio,
                            "log_mel": log_mel.cpu() if hasattr(log_mel, "cpu") else log_mel,
                            "mel": mel.cpu() if hasattr(mel, "cpu") else mel,
                            "mel_len": mel_len,
                            "spk_emb": spk_emb,
                        })
                        # persist speaker embedding and features for future sessions
                        try:
                            PROMPT_FEATURE_CACHE.persist_entry(prompt_wav)
                        except Exception:
                            pass
                    except Exception:
                        # Avoid failing dataset construction if cache save fails
                        pass
                
                # 4. feature for llm
                prompt_text = normalize_text(prompt_text) # remove some space and strange character
                prompt_text = f"{SPK_DICT[spk_idx]}{TEXT_START}{prompt_text}{TEXT_END}{AUDIO_START}"
                if spk_idx == 0:
                    prompt_text = f"{TASK_PODCAST}{prompt_text}"
                prompt_text_ids = self.text_tokenizer.encode(prompt_text)
                prompt_text_ids_list.append(prompt_text_ids)
                if use_dialect_prompt:
                    dialect_prompt_text = normalize_text(data["dialect_prompt_text"][spk_idx])
                    dialect_prompt_text = f"{SPK_DICT[spk_idx]}{TEXT_START}{dialect_prompt_text}{TEXT_END}{AUDIO_START}"
                    dialect_prompt_text_ids = self.text_tokenizer.encode(dialect_prompt_text)
                    dialect_prompt_text_ids_list.append(dialect_prompt_text_ids)
                    if spk_idx == 0:
                        dialect_prefix_list.append(self.text_tokenizer.encode(f"{TASK_PODCAST}"))
                    else:
                        dialect_prefix_list.append([])
                log_mel_list.append(log_mel)
                spk_emb_list.append(spk_emb)
                mel_list.append(mel); mel_len_list.append(mel_len)
            item = {
                "prompt_text_tokens": prompt_text_ids_list,
                "spk_emb": spk_emb_list, "mel": mel_list, "mel_len": mel_len_list, "log_mel": log_mel_list, "info": data,
            }
            if use_dialect_prompt:
                item.update({
                    "use_dialect_prompt": True,
                    "dialect_prompt_text_tokens": dialect_prompt_text_ids_list,
                    "dialect_prefix": dialect_prefix_list,
                })
            text_ids_list, spks_list = [], []
            if "spk" not in data:
                data["spk"] = [0] * len(data["text"])

            for text, spk in zip(data["text"], data["spk"]):
                # 4. feature for llm
                text = normalize_text(text)
                text = f"{SPK_DICT[spk]}{TEXT_START}{text}{TEXT_END}{AUDIO_START}"
                text_ids = self.text_tokenizer.encode(text)

                text_ids_list.append(text_ids)
                spks_list.append(spk)

            item.update({
                "text_tokens": text_ids_list, "spks_list": spks_list,
            })
        except Exception as e:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [WARNING] - Error processing data item {data.get('key', idx)}: {e}")
            return None
        return item

class PodcastInferHandler(PodcastDataset):

    def __init__(self, text_tokenizer, data_list, model_config: Config):
        self.datas = []
        self.model_config = model_config

        """Example data_list:
        ```
        {"key": "uttid_1", "prompt_text": ["prompt_text1", "prompt_text2"], "dialect_prompt_text": ["dialect_prompt_text1", "dialect_prompt_text2"], "text": ["text1", "text2], "spk": [0, 1], "prompt_wav": ["/mnt/data/audio/00000000.wav", "/mnt/data/audio/00000001.wav"], "wav": "/mnt/data/audio_synthetic/uttid_1.wav"}
        ```
        Note:
            - `key` is the key of this sample.
            - `prompt_text` is the text used for prompt.
            - `dialect_prompt_text` is the cot text used for prompt as to activate specific ability.
            - `text` is the text used for generating real audio.
            - `spk` is the target speaker id to synthesize, corresponds to the prompt order. Default SPEAKER_0.
            - `prompt_wav` is the audio used for prompt.
            - `wav` is the path to the generated audio to be saved (we highly recommend to pre-define the save path before running the script).
        """
        missing = 0
        self.text_tokenizer = text_tokenizer

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(f"{self.model_config.model}/campplus.onnx", sess_options=option,
                                                    providers=["CPUExecutionProvider"])

    def update_datasource(self, data_list):
        self.datas = data_list