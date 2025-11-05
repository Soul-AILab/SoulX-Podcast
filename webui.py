import re
import gradio as gr
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Literal, List, Tuple
import sys
import importlib.util
from datetime import datetime

import torch
import numpy as np  
import random    
import s3tokenizer

from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams
from soulxpodcast.utils.dataloader import (
    PodcastInferHandler,
    SPK_DICT, TEXT_START, TEXT_END, AUDIO_START, TASK_PODCAST
)


S1_PROMPT_WAV = "example/audios/female_mandarin.wav"  
S2_PROMPT_WAV = "example/audios/male_mandarin.wav"  

def load_dialect_prompt_data():
    """
    加载方言提示文本文件并格式化为嵌套字典。
    返回结构: {dialect_key: {display_name: full_text, ...}, ...}
    """
    dialect_data = {}
    
    dialect_files = [
        ("sichuan", "example/dialect_prompt/sichuan.txt", "<|Sichuan|>"),
        ("yueyu", "example/dialect_prompt/yueyu.txt", "<|Yue|>"),
        ("henan", "example/dialect_prompt/henan.txt", "<|Henan|>"),
    ]
    
    for key, file_path, prefix in dialect_files:
        dialect_data[key] = {"(无)": ""} 
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        full_text = f"{prefix}{line}"
                        display_name = f"例{i+1}: {line[:20]}..."
                        dialect_data[key][display_name] = full_text
        except FileNotFoundError:
            print(f"[WARNING] 方言文件未找到: {file_path}")
        except Exception as e:
            print(f"[WARNING] 读取方言文件失败 {file_path}: {e}")
            
    return dialect_data

DIALECT_PROMPT_DATA = load_dialect_prompt_data()
DIALECT_CHOICES = ["(无)", "sichuan", "yueyu", "henan"]


EXAMPLES_LIST = [
    [
        None, "", "", None, "", "", None, "", "", None, "", "", ""
    ],
    [
        S1_PROMPT_WAV,
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
        "",
        S2_PROMPT_WAV,
        "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
        "",
        None, "", "",
        None, "", "",
        "[S1] 哈喽，AI时代的冲浪先锋们！欢迎收听《AI生活进行时》。啊，一个充满了未来感，然后，还有一点点，<|laughter|>神经质的播客节目，我是主持人小希。\n[S2] 哎，大家好呀！我是能唠，爱唠，天天都想唠的唠嗑！\n[S1] 最近活得特别赛博朋克哈！以前老是觉得AI是科幻片儿里的，<|sigh|> 现在，现在连我妈都用AI写广场舞文案了。\n[S2] 这个例子很生动啊。是的，特别是生成式AI哈，感觉都要炸了！ 诶，那我们今天就聊聊AI是怎么走进我们的生活的哈！",
    ],
    [
        S1_PROMPT_WAV,
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
        "<|Sichuan|>要得要得！前头几个耍洋盘，我后脚就背起铺盖卷去景德镇耍泥巴，巴适得喊老天爷！",
        S2_PROMPT_WAV,
        "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
        "<|Sichuan|>哎哟喂，这个搞反了噻！黑神话里头唱曲子的王二浪早八百年就在黄土高坡吼秦腔喽，游戏组专门跑切录的原汤原水，听得人汗毛儿都立起来！",
        None, "", "",
        None, "", "",
        "[S1] <|Sichuan|>各位《巴适得板》的听众些，大家好噻！我是你们主持人晶晶。今儿天气硬是巴适，不晓得大家是在赶路嘛，还是茶都泡起咯，准备跟我们好生摆一哈龙门阵喃？\n[S2] <|Sichuan|>晶晶好哦，大家安逸噻！我是李老倌。你刚开口就川味十足，摆龙门阵几个字一甩出来，我鼻子头都闻到茶香跟火锅香咯！\n[S1] <|Sichuan|>就是得嘛！李老倌，我前些天带个外地朋友切人民公园鹤鸣茶社坐了一哈。他硬是搞不醒豁，为啥子我们一堆人围到杯茶就可以吹一下午壳子，从隔壁子王嬢嬢娃儿耍朋友，扯到美国大选，中间还掺几盘斗地主。他说我们四川人简直是把摸鱼刻进骨子里头咯！\n[S2] <|Sichuan|>你那个朋友说得倒是有点儿趣，但他莫看到精髓噻。摆龙门阵哪是摸鱼嘛，这是我们川渝人特有的交际方式，更是一种活法。外省人天天说的松弛感，根根儿就在这龙门阵里头。今天我们就要好生摆一哈，为啥子四川人活得这么舒坦。就先从茶馆这个老窝子说起，看它咋个成了我们四川人的魂儿！",
    ],
    [
        S1_PROMPT_WAV,
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
        "<|Yue|>真係冇讲错啊！攀山滑雪嘅语言专家几巴闭，都唔及我听日拖成副身家去景德镇玩泥巴，呢铺真系发哂白日梦咯！",
        S2_PROMPT_WAV,
        "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
        "<|Yue|>咪搞错啊！陕北民谣响度唱咗几十年，黑神话边有咁大面啊？你估佢哋抄游戏咩！",
        None, "", "",
        None, "", "",
        "[S1] <|Yue|>哈囉大家好啊，歡迎收聽我哋嘅節目。喂，我今日想問你樣嘢啊，你覺唔覺得，嗯，而家揸電動車，最煩，最煩嘅一樣嘢係咩啊？\n[S2] <|Yue|>梗係充電啦。大佬啊，搵個位都已經好煩，搵到個位仲要喺度等，你話快極都要半個鐘一個鐘，真係，有時諗起都覺得好冇癮。\n[S1] <|Yue|>係咪先。如果我而家同你講，充電可以快到同入油差唔多時間，你信唔信先？喂你平時喺油站入滿一缸油，要幾耐啊？五六分鐘？\n[S2] <|Yue|>差唔多啦，七八分鐘，點都走得啦。電車喎，可以做到咁快？你咪玩啦。",
    ],
    [
        S1_PROMPT_WAV,
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
        "<|Henan|>俺这不是怕恁路上不得劲儿嘛！那景德镇瓷泥可娇贵着哩，得先拿咱河南人这实诚劲儿给它揉透喽。",
        S2_PROMPT_WAV,
        "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
        "<|Henan|>恁这想法真闹挺！陕北民谣比黑神话早几百年都有了，咱可不兴这弄颠倒啊，中不？恁这想法真闹挺！那陕北民谣在黄土高坡响了几百年，咋能说是跟黑神话学的咧？咱得把这事儿捋直喽，中不中！",
        None, "", "",
        None, "", "",
        "[S1] <|Henan|>哎，大家好啊，欢迎收听咱这一期嘞《瞎聊呗，就这么说》，我是恁嘞老朋友，燕子。\n[S2] <|Henan|>大家好，我是老张。燕子啊，今儿瞅瞅你这个劲儿，咋着，是有啥可得劲嘞事儿想跟咱唠唠？\n[S1] <|Henan|>哎哟，老张，你咋恁懂我嘞！我跟你说啊，最近我刷手机，老是刷住些可逗嘞方言视频，特别是咱河南话，咦～我哩个乖乖，一听我都憋不住笑，咋说嘞，得劲儿哩很，跟回到家一样。\n[S2] <|Henan|>你这回可算说到根儿上了！河南话，咱往大处说说，中原官话，它真嘞是有一股劲儿搁里头。它可不光是说话，它脊梁骨后头藏嘞，是咱一整套、鲜鲜活活嘞过法儿，一种活人嘞道理。\n[S1] <|Henan|>活人嘞道理？哎，这你这一说，我嘞兴致'腾'一下就上来啦！觉住咱这嗑儿，一下儿从搞笑视频蹿到文化顶上了。那你赶紧给我白话白话，这里头到底有啥道道儿？我特别想知道——为啥一提起咱河南人，好些人脑子里'蹦'出来嘞头一个词儿，就是实在？这个实在，骨子里到底是啥嘞？",
    ],
]


model: SoulXPodcast = None
dataset: PodcastInferHandler = None
def initiate_model(config: Config, enable_tn: bool=False):
    global model
    if model is None:
        model = SoulXPodcast(config)

    global dataset
    if dataset is None:
        dataset = PodcastInferHandler(model.llm.tokenizer, None, config)

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
)


global_lang: Literal["zh", "en"] = "zh"

def i18n(key):
    global global_lang
    return _i18n_key2lang_dict[key][global_lang]

def check_monologue_text(text: str, prefix: str = None) -> bool:
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
    if len(text_list) == 0:
        return False
    for text in text_list:
        # 检查是否匹配 [S1] 到 [S{max_speakers}] 格式
        pattern = r'^\[S([1-9]|[1-9][0-9]+)\].*'
        match = re.match(pattern, text.strip())
        if not match:
            return False
        spk_num = int(match.group(1))
        if spk_num < 1:
            return False
        if max_speakers is not None and spk_num > max_speakers:
            return False
    return True

def process_single(target_text_list, prompt_wav_list, prompt_text_list, use_dialect_prompt, dialect_prompt_text):
    spks, texts = [], []
    for target_text in target_text_list:
        pattern = r'(\[S([1-9]|[1-9][0-9]+)\])(.+)'
        match = re.match(pattern, target_text)
        if not match:
            continue
        spk_num = int(match.group(2))
        text = match.group(3).strip()
        spk = spk_num - 1  # S1->0, S2->1, etc.
        spks.append(spk)
        texts.append(text)
    
    global dataset
    dataitem = {"key": "001", "prompt_text": prompt_text_list, "prompt_wav": prompt_wav_list, 
             "text": texts, "spk": spks, }
    if use_dialect_prompt:
        dataitem.update({
            "dialect_prompt_text": dialect_prompt_text
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


def dialogue_synthesis_function(
    target_text: str,
    speaker_configs_list: List[Tuple[str, str, str]],  # List of (prompt_text, prompt_audio, dialect_prompt_text)
    seed: int = 1988,
):
    """
    合成对话音频
    speaker_configs_list: 说话人配置列表，每个元素为 (prompt_text, prompt_audio, dialect_prompt_text)
    """
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Check prompt info
    # 匹配 [S1]... 到下一个 [Sx] 或文本结尾
    # 使用非贪婪匹配，允许中间包含其他方括号标签（如 [laughter], [breath] 等）
    pattern = r'\[S([1-9]|[1-9][0-9]+)\](.*?)(?=\[S([1-9]|[1-9][0-9]+)\]|$)'
    matches = list(re.finditer(pattern, target_text, re.DOTALL))
    # 重新组合完整匹配：说话人标签 + 内容
    target_text_list: List[str] = []
    for match in matches:
        spk_num = match.group(1)  # 说话人编号
        content = match.group(2)  # 内容部分
        print(f"spk_num: {spk_num}, content: {content}")
        # 重新组合为 [S1]内容 的格式
        full_text = f"[S{spk_num}]{content}".strip()
        target_text_list.append(full_text)
    
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
    progress_bar = gr.Progress(track_tqdm=True)
    prompt_wav_list = [config[1] for config in speaker_configs_list[:max_spk_used]]
    prompt_text_list = [config[0] for config in speaker_configs_list[:max_spk_used]]
    use_dialect_prompt = any(config[2].strip() != "" for config in speaker_configs_list[:max_spk_used])
    dialect_prompt_text_list = [config[2] for config in speaker_configs_list[:max_spk_used]]
    data = process_single(
        target_text_list,
        prompt_wav_list,
        prompt_text_list,
        use_dialect_prompt,
        dialect_prompt_text_list,
    )
    results_dict = model.forward_longform(
        **data
    )
    target_audio = None
    for i in range(len(results_dict['generated_wavs'])):
        if target_audio is None:
            target_audio = results_dict['generated_wavs'][i]
        else:
            target_audio = torch.concat([target_audio, results_dict['generated_wavs'][i]], axis=1)
    return (24000, target_audio.cpu().squeeze(0).numpy())


def update_example_choices(dialect_key: str):

    if dialect_key == "(无)":
        choices = ["(请先选择方言)"]

        return gr.update(choices=choices, value="(无)"), gr.update(choices=choices, value="(无)")
    
    choices = list(DIALECT_PROMPT_DATA.get(dialect_key, {}).keys())

    return gr.update(choices=choices, value="(无)"), gr.update(choices=choices, value="(无)")

def update_prompt_text(dialect_key: str, example_key: str):
    if dialect_key == "(无)" or example_key in ["(无)", "(请先选择方言)"]:
        return gr.update(value="")
    

    full_text = DIALECT_PROMPT_DATA.get(dialect_key, {}).get(example_key, "")
    return gr.update(value=full_text)


def create_speaker_group(spk_num: int):
    """创建一个说话人组件组"""
    with gr.Group(visible=True) as group:
        # 添加复选框用于选择删除
        checkbox = gr.Checkbox(
            label=f"选择说话人 {spk_num}",
            value=False,
            scale=0,
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
    return group, checkbox, prompt_audio, prompt_text, dialect_prompt_text


def render_interface() -> gr.Blocks:
    with gr.Blocks(title="SoulX-Podcast", theme=gr.themes.Default()) as page:

        with gr.Row():
            lang_choice = gr.Radio(
                choices=["中文", "English"],
                value="中文",
                label="Display Language/显示语言",
                type="index",
                interactive=True,
                scale=3,
            )
            seed_input = gr.Number(
                label="Seed (种子)",
                value=1988,
                step=1,
                interactive=True,
                scale=1,
            )

        # 说话人状态管理（最多支持10个说话人）
        MAX_SPEAKERS = 10
        speakers_state = gr.State(value=1)  # 当前说话人数量
        
        # 创建所有说话人组件（最多10个）
        speaker_checkbox_list = []
        speaker_audio_list = []
        speaker_text_list = []
        speaker_dialect_list = []
        speaker_columns = []
        
        with gr.Row() as speakers_row:
            for i in range(MAX_SPEAKERS):
                with gr.Column(scale=1, visible=(i < 1)) as col:
                    group, checkbox, audio, text, dialect = create_speaker_group(i + 1)
                    speaker_checkbox_list.append(checkbox)
                    speaker_audio_list.append(audio)
                    speaker_text_list.append(text)
                    speaker_dialect_list.append(dialect)
                    speaker_columns.append(col)
        
        # 添加/删除说话人按钮
        with gr.Row():
            add_speaker_btn = gr.Button("➕ 添加1个说话人", variant="secondary", scale=1)
            with gr.Group():
                quick_add_num = gr.Number(
                    label="快速添加数量",
                    value=1,
                    minimum=1,
                    maximum=MAX_SPEAKERS,
                    step=1,
                    precision=0,
                    scale=1,
                )
                quick_add_btn = gr.Button("🚀 快速添加", variant="primary", scale=1)
            select_all_btn = gr.Button("☑️ 全选", variant="secondary", scale=0)
            select_none_btn = gr.Button("☐ 全不选", variant="secondary", scale=0)
            batch_delete_btn = gr.Button("🗑️ 批量删除选中", variant="stop", scale=1)
        
        def update_speakers_visibility(num_speakers):
            """更新说话人列的可见性和标签"""
            updates = []
            for i in range(MAX_SPEAKERS):
                visible = (i < num_speakers)
                if visible:
                    # 更新复选框标签
                    updates.append(gr.update(visible=True, label=f"选择说话人 {i + 1}", value=False))
                else:
                    updates.append(gr.update(visible=False, value=False))
            return updates
        
        def add_speaker(current_num):
            """添加一个说话人"""
            new_num = min(current_num + 1, MAX_SPEAKERS)
            checkbox_updates = update_speakers_visibility(new_num)
            column_updates = [gr.update(visible=(i < new_num)) for i in range(MAX_SPEAKERS)]
            return new_num, *checkbox_updates, *column_updates
        
        def quick_add_speakers(current_num, add_count):
            """快速添加指定数量的说话人"""
            add_count = int(add_count) if add_count else 1
            add_count = max(1, min(add_count, MAX_SPEAKERS - current_num))  # 确保不超过最大值
            new_num = min(current_num + add_count, MAX_SPEAKERS)
            checkbox_updates = update_speakers_visibility(new_num)
            column_updates = [gr.update(visible=(i < new_num)) for i in range(MAX_SPEAKERS)]
            return new_num, *checkbox_updates, *column_updates
        
        def batch_delete_speakers(current_num, *all_values):
            """批量删除选中的说话人，并重新排列剩余说话人及其数据"""
            # 分离复选框值和其他数据
            # all_values格式: (checkbox1, audio1, text1, dialect1, checkbox2, audio2, text2, dialect2, ...)
            checkbox_values = []
            audio_values = []
            text_values = []
            dialect_values = []
            
            for i in range(MAX_SPEAKERS):
                idx = i * 4
                if idx < len(all_values):
                    checkbox_values.append(all_values[idx])
                    if idx + 1 < len(all_values):
                        audio_values.append(all_values[idx + 1])
                    if idx + 2 < len(all_values):
                        text_values.append(all_values[idx + 2])
                    if idx + 3 < len(all_values):
                        dialect_values.append(all_values[idx + 3])
            
            # 找出所有选中的说话人索引
            selected_indices = set([i for i, checked in enumerate(checkbox_values) if checked and i < current_num])
            
            if not selected_indices:
                gr.Warning("请至少选择一个说话人进行删除")
                checkbox_updates = update_speakers_visibility(current_num)
                # 返回所有组件（复选框、音频、文本、方言）的更新，保持原值不变
                result = []
                for i in range(MAX_SPEAKERS):
                    result.append(checkbox_updates[i])  # checkbox
                    result.append(gr.update())  # audio - 保持原值
                    result.append(gr.update())  # text - 保持原值
                    result.append(gr.update())  # dialect - 保持原值
                column_updates = [gr.update(visible=(i < current_num)) for i in range(MAX_SPEAKERS)]
                return current_num, *result, *column_updates
            
            # 检查是否会删除所有说话人
            remaining_count = current_num - len(selected_indices)
            if remaining_count < 1:
                gr.Warning("至少需要保留1个说话人")
                checkbox_updates = update_speakers_visibility(current_num)
                result = []
                for i in range(MAX_SPEAKERS):
                    result.append(checkbox_updates[i])
                    result.append(gr.update())  # audio - 保持原值
                    result.append(gr.update())  # text - 保持原值
                    result.append(gr.update())  # dialect - 保持原值
                column_updates = [gr.update(visible=(i < current_num)) for i in range(MAX_SPEAKERS)]
                return current_num, *result, *column_updates
            
            # 找出保留的说话人索引
            kept_indices = [i for i in range(current_num) if i not in selected_indices]
            new_num = remaining_count
            
            # 重新排列数据：将保留的说话人数据移到前面
            result = []
            for i in range(MAX_SPEAKERS):
                if i < new_num:
                    # 保留的说话人，从kept_indices[i]位置取数据
                    old_idx = kept_indices[i]
                    # 更新复选框
                    result.append(gr.update(visible=True, label=f"选择说话人 {i + 1}", value=False))
                    # 更新音频（如果原位置有值则使用，否则为None）
                    audio_val = audio_values[old_idx] if old_idx < len(audio_values) else None
                    result.append(gr.update(value=audio_val))
                    # 更新文本
                    text_val = text_values[old_idx] if old_idx < len(text_values) else ""
                    result.append(gr.update(value=text_val))
                    # 更新方言
                    dialect_val = dialect_values[old_idx] if old_idx < len(dialect_values) else ""
                    result.append(gr.update(value=dialect_val))
                else:
                    # 隐藏的说话人，清空数据
                    result.append(gr.update(visible=False, value=False))  # checkbox
                    result.append(gr.update(value=None))  # audio
                    result.append(gr.update(value=""))  # text
                    result.append(gr.update(value=""))  # dialect
            
            # 列的更新
            column_updates = [gr.update(visible=(i < new_num)) for i in range(MAX_SPEAKERS)]
            
            return new_num, *result, *column_updates
        
        add_speaker_btn.click(
            fn=add_speaker,
            inputs=[speakers_state],
            outputs=[speakers_state] + speaker_checkbox_list + speaker_columns
        )
        
        quick_add_btn.click(
            fn=quick_add_speakers,
            inputs=[speakers_state, quick_add_num],
            outputs=[speakers_state] + speaker_checkbox_list + speaker_columns
        )
        
        def select_all_checkboxes(current_num):
            """全选所有可见的复选框"""
            updates = []
            for i in range(MAX_SPEAKERS):
                if i < current_num:
                    updates.append(gr.update(value=True))
                else:
                    updates.append(gr.update())
            return updates
        
        def select_none_checkboxes(current_num):
            """取消全选所有复选框"""
            updates = []
            for i in range(MAX_SPEAKERS):
                updates.append(gr.update(value=False))
            return updates
        
        select_all_btn.click(
            fn=select_all_checkboxes,
            inputs=[speakers_state],
            outputs=speaker_checkbox_list
        )
        
        select_none_btn.click(
            fn=select_none_checkboxes,
            inputs=[speakers_state],
            outputs=speaker_checkbox_list
        )
        
        # 准备所有输入组件（复选框、音频、文本、方言）
        all_speaker_inputs_for_delete = []
        for i in range(MAX_SPEAKERS):
            all_speaker_inputs_for_delete.extend([
                speaker_checkbox_list[i],
                speaker_audio_list[i],
                speaker_text_list[i],
                speaker_dialect_list[i]
            ])
        
        # 准备所有输出组件（复选框、音频、文本、方言）
        all_speaker_outputs_for_delete = []
        for i in range(MAX_SPEAKERS):
            all_speaker_outputs_for_delete.extend([
                speaker_checkbox_list[i],
                speaker_audio_list[i],
                speaker_text_list[i],
                speaker_dialect_list[i]
            ])
        
        batch_delete_btn.click(
            fn=batch_delete_speakers,
            inputs=[speakers_state] + all_speaker_inputs_for_delete,
            outputs=[speakers_state] + all_speaker_outputs_for_delete + speaker_columns
        )

        with gr.Row():
            with gr.Column(scale=1):
                dialogue_text_input = gr.Textbox(
                    label=i18n("dialogue_text_input_label"),
                    placeholder=i18n("dialogue_text_input_placeholder"),
                    lines=18,
                )

        # Generate button
        with gr.Row():
            generate_btn = gr.Button(
                value=i18n("generate_btn_label"), 
                variant="primary", 
                scale=3,
                size="lg",
            )
        
        # Long output audio
        generate_audio = gr.Audio(
            label=i18n("generated_audio_label"),
            interactive=False,
        )


        # 收集说话人配置的包装函数
        def collect_and_synthesize(target_text, num_speakers, seed, *speaker_args):
            """收集所有说话人配置并调用合成函数"""
            # speaker_args格式: (audio1, text1, dialect1, audio2, text2, dialect2, ...)
            # 只收集可见的说话人（前num_speakers个）
            speaker_configs = []
            num = int(num_speakers)
            for i in range(0, min(num * 3, len(speaker_args)), 3):
                if i + 2 < len(speaker_args):
                    audio = speaker_args[i] if speaker_args[i] is not None else None
                    text = speaker_args[i+1] if speaker_args[i+1] is not None else ""
                    dialect = speaker_args[i+2] if speaker_args[i+2] is not None else ""
                    speaker_configs.append((text, audio, dialect))
            return dialogue_synthesis_function(target_text, speaker_configs, seed)
        
        # 生成按钮点击事件
        all_speaker_inputs = []
        for i in range(MAX_SPEAKERS):
            all_speaker_inputs.extend([
                speaker_audio_list[i],
                speaker_text_list[i],
                speaker_dialect_list[i]
            ])
        
        generate_btn.click(
            fn=collect_and_synthesize,
            inputs=[
                dialogue_text_input,
                speakers_state,
                seed_input,
                *all_speaker_inputs,
            ],
            outputs=[generate_audio],
        )
        
        # 语言切换
        def _change_component_language(lang):
            global global_lang
            global_lang = ["zh", "en"][lang]
            updates = []
            # 更新所有说话人组件
            for i in range(MAX_SPEAKERS):
                updates.extend([
                    gr.update(label=i18n(f"spk{i+1}_prompt_audio_label") if f"spk{i+1}_prompt_audio_label" in _i18n_key2lang_dict else f"说话人 {i+1} 参考语音"),
                    gr.update(
                        label=i18n(f"spk{i+1}_prompt_text_label") if f"spk{i+1}_prompt_text_label" in _i18n_key2lang_dict else f"说话人 {i+1} 参考文本",
                        placeholder=i18n(f"spk{i+1}_prompt_text_placeholder") if f"spk{i+1}_prompt_text_placeholder" in _i18n_key2lang_dict else f"说话人 {i+1} 参考文本",
                    ),
                    gr.update(
                        label=i18n(f"spk{i+1}_dialect_prompt_text_label") if f"spk{i+1}_dialect_prompt_text_label" in _i18n_key2lang_dict else f"说话人 {i+1} 方言提示文本",
                        placeholder=i18n(f"spk{i+1}_dialect_prompt_text_placeholder") if f"spk{i+1}_dialect_prompt_text_placeholder" in _i18n_key2lang_dict else "带前缀方言提示词思维链文本",
                    ),
                ])
            # 添加对话文本、生成按钮和音频输出
            updates.extend([
                gr.update(
                    label=i18n("dialogue_text_input_label"),
                    placeholder=i18n("dialogue_text_input_placeholder"),
                ),
                gr.update(value=i18n("generate_btn_label")),
                gr.update(label=i18n("generated_audio_label")),
            ])
            return updates
        
        lang_choice.change(
            fn=_change_component_language,
            inputs=[lang_choice],
            outputs=all_speaker_inputs + [dialogue_text_input, generate_btn, generate_audio],
        )
    return page


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model_path',
                        required=True,
                        type=str,
                        help='model path')
    parser.add_argument('--llm_engine',
                        type=str,
                        default="hf",
                        help='model execute engine')
    parser.add_argument('--fp16_flow',
                        action='store_true',
                        help='enable fp16 flow')
    parser.add_argument('--seed',
                        type=int,
                        default=1988,
                        help='random seed for generation')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Initiate model
    hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
            initial_values={"fp16_flow": args.fp16_flow}, 
            json_file=f"{args.model_path}/soulxpodcast_config.json")
    
    llm_engine = args.llm_engine
    if llm_engine == "vllm":
        if not importlib.util.find_spec("vllm"):
            llm_engine = "hf"
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [WARNING]: No install VLLM, switch to hf engine.")
    config = Config(model=args.model_path, enforce_eager=True, llm_engine=llm_engine,
                    hf_config=hf_config)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    initiate_model(config)
    print("[INFO] SoulX-Podcast loaded")    
    page = render_interface()
    page.queue()
    page.launch(share=False)
