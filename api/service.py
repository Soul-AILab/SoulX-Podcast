"""
SoulXPodcast Model Service Layer
"""
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import numpy as np
import random
import gc
import threading

from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams
from soulxpodcast.utils.dataloader import PodcastInferHandler

from api.config import config as api_config
from api.utils import parse_dialogue_text

logger = logging.getLogger(__name__)


class SoulXPodcastService:
    """SoulXPodcast模型服务单例"""

    _instance: Optional['SoulXPodcastService'] = None
    _lock = threading.Lock()  # 线程锁

    def __new__(cls):
        with cls._lock:  # 确保线程安全
            if cls._instance is None:
                logger.info("Creating new SoulXPodcastService instance")
                cls._instance = super(SoulXPodcastService, cls).__new__(cls)
                cls._instance._initialized = False  # 实例属性
                cls._instance._generation_lock = threading.Lock()  # 生成锁
        return cls._instance

    def __init__(self):
        """初始化模型（单例模式，只初始化一次）"""
        # 使用实例属性而不是类属性
        if not self._initialized:
            logger.info("Initializing SoulXPodcastService (first time)")
            self._load_model()
            self._initialized = True  # 设置实例属性
        else:
            logger.info("SoulXPodcastService already initialized, skipping model load")

    def _load_model(self):
        """加载模型"""
        try:
            logger.info(f"Loading SoulXPodcast model from {api_config.model_path}...")
            logger.info(f"Using LLM engine: {api_config.llm_engine}")

            # 加载配置
            hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
                initial_values={"fp16_flow": api_config.fp16_flow},
                json_file=f"{api_config.model_path}/soulxpodcast_config.json"
            )

            # 创建Config对象
            model_config = Config(
                model=api_config.model_path,
                enforce_eager=True,
                llm_engine=api_config.llm_engine,
                hf_config=hf_config
            )

            # 初始化模型
            self.model = SoulXPodcast(model_config)
            self.dataset = PodcastInferHandler(
                self.model.llm.tokenizer,
                None,
                model_config
            )
            self.config = model_config

            logger.info(f"Model loaded successfully with {api_config.llm_engine} engine!")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return hasattr(self, 'model') and self.model is not None

    def generate(
        self,
        prompt_audio_paths: List[str],
        prompt_texts: List[str],
        dialogue_text: str,
        seed: int = 1988,
        temperature: float = 0.6,
        top_k: int = 100,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
    ) -> Tuple[int, np.ndarray]:
        """
        生成语音

        Args:
            prompt_audio_paths: 参考音频路径列表
            prompt_texts: 参考文本列表
            dialogue_text: 对话文本
            seed: 随机种子
            temperature: 采样温度
            top_k: Top-K采样
            top_p: Top-P采样
            repetition_penalty: 重复惩罚

        Returns:
            Tuple[int, np.ndarray]: (采样率, 音频数组)
        """
        logger.info(f"Generate called - Instance ID: {id(self)}, Model loaded: {self.is_loaded()}")

        if not self.is_loaded():
            raise RuntimeError("模型未加载")

        # 使用锁确保同一时间只有一个生成任务
        with self._generation_lock:
            logger.info("Acquired generation lock")
            try:
                # 设置随机种子
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                num_speakers = len(prompt_audio_paths)
                logger.info(f"Generating audio for {num_speakers} speaker(s)")

                # 解析对话文本
                target_text_list = parse_dialogue_text(dialogue_text, num_speakers)
                logger.info(f"Parsed dialogue into {len(target_text_list)} segments")

                # 提取说话人和文本
                spks, texts = [], []
                for target_text in target_text_list:
                    pattern = r'(\[S[1-9]\])(.+)'
                    match = re.match(pattern, target_text)
                    if match:
                        text, spk = match.group(2), int(match.group(1)[2]) - 1
                        spks.append(spk)
                        texts.append(text)
                    else:
                        raise ValueError(f"无效的对话文本格式: {target_text}")

                # 构建数据项
                dataitem = {
                    "key": "api_001",
                    "prompt_text": prompt_texts,
                    "prompt_wav": prompt_audio_paths,
                    "text": texts,
                    "spk": spks,
                }

                # 更新数据源
                self.dataset.update_datasource([dataitem])

                # 获取处理后的数据
                data = self.dataset[0]

                # 准备模型输入
                import s3tokenizer
                prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(data["log_mel"])
                spk_emb_for_flow = torch.tensor(data["spk_emb"])
                prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(
                    data["mel"], batch_first=True, padding_value=0
                )
                prompt_mels_lens_for_flow = torch.tensor(data['mel_len'])
                text_tokens_for_llm = data["text_tokens"]
                prompt_text_tokens_for_llm = data["prompt_text_tokens"]
                spk_ids = data["spks_list"]

                # 采样参数
                sampling_params = SamplingParams(
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                    top_p=top_p,
                    use_ras=True,
                    win_size=25,
                    tau_r=0.2
                )

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
                    "use_dialect_prompt": False,
                }

                # 模型推理
                logger.info("Running model inference...")

                # 清理之前可能累积的GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 使用超时机制执行推理
                import concurrent.futures
                import signal

                def run_inference():
                    """在独立线程中执行推理"""
                    with torch.no_grad():
                        return self.model.forward_longform(**processed_data)

                # 设置超时时间（根据音频长度动态调整）
                num_segments = len(texts)
                timeout_seconds = max(1200, num_segments * 120)  # 每段至少120秒，最少20分钟(1200秒)

                logger.info(f"Starting inference with timeout: {timeout_seconds}s for {num_segments} segments")

                # 使用线程池执行推理
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_inference)
                    try:
                        results_dict = future.result(timeout=timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Model inference timeout after {timeout_seconds} seconds")
                        # 尝试取消任务
                        future.cancel()
                        # 清理GPU内存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise TimeoutError(f"模型推理超时（{timeout_seconds}秒）。可能是音频过长或GPU内存不足。")
                    except Exception as e:
                        logger.error(f"Model inference failed: {e}")
                        raise RuntimeError(f"模型推理失败: {str(e)}")

                # 拼接音频
                target_audio = None
                for i in range(len(results_dict['generated_wavs'])):
                    if target_audio is None:
                        target_audio = results_dict['generated_wavs'][i]
                    else:
                        target_audio = torch.concat(
                            [target_audio, results_dict['generated_wavs'][i]], axis=1
                        )

                # 转换为numpy数组
                audio_array = target_audio.cpu().squeeze(0).numpy()
                sample_rate = 24000

                # 清理GPU内存
                del target_audio
                del results_dict
                if 'processed_data' in locals():
                    del processed_data

                # 显式清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 强制垃圾回收
                gc.collect()

                logger.info(f"Audio generation completed. Duration: {len(audio_array) / sample_rate:.2f}s")

                # 记录GPU内存使用情况
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                    logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

                return sample_rate, audio_array

            except Exception as e:
                logger.error(f"Generation failed: {e}", exc_info=True)
                raise RuntimeError(f"语音生成失败: {str(e)}")
            finally:
                logger.info("Released generation lock")


# 全局服务实例
_service: Optional[SoulXPodcastService] = None


def get_service() -> SoulXPodcastService:
    """获取全局服务实例"""
    global _service
    if _service is None:
        _service = SoulXPodcastService()
    return _service
