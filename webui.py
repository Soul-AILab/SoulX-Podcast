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
        None, "", "", None, "", "", ""
    ],
    [
        S1_PROMPT_WAV,
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
        "",
        S2_PROMPT_WAV,
        "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
        "",
        "[S1] 哈喽，AI时代的冲浪先锋们！欢迎收听《AI生活进行时》。啊，一个充满了未来感，然后，还有一点点，<|laughter|>神经质的播客节目，我是主持人小希。\n[S2] 哎，大家好呀！我是能唠，爱唠，天天都想唠的唠嗑！\n[S1] 最近活得特别赛博朋克哈！以前老是觉得AI是科幻片儿里的，<|sigh|> 现在，现在连我妈都用AI写广场舞文案了。\n[S2] 这个例子很生动啊。是的，特别是生成式AI哈，感觉都要炸了！ 诶，那我们今天就聊聊AI是怎么走进我们的生活的哈！",
    ],
    [
        S1_PROMPT_WAV,
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
        "<|Sichuan|>要得要得！前头几个耍洋盘，我后脚就背起铺盖卷去景德镇耍泥巴，巴适得喊老天爷！",
        S2_PROMPT_WAV,
        "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
        "<|Sichuan|>哎哟喂，这个搞反了噻！黑神话里头唱曲子的王二浪早八百年就在黄土高坡吼秦腔喽，游戏组专门跑切录的原汤原水，听得人汗毛儿都立起来！",
        "[S1] <|Sichuan|>各位《巴适得板》的听众些，大家好噻！我是你们主持人晶晶。今儿天气硬是巴适，不晓得大家是在赶路嘛，还是茶都泡起咯，准备跟我们好生摆一哈龙门阵喃？\n[S2] <|Sichuan|>晶晶好哦，大家安逸噻！我是李老倌。你刚开口就川味十足，摆龙门阵几个字一甩出来，我鼻子头都闻到茶香跟火锅香咯！\n[S1] <|Sichuan|>就是得嘛！李老倌，我前些天带个外地朋友切人民公园鹤鸣茶社坐了一哈。他硬是搞不醒豁，为啥子我们一堆人围到杯茶就可以吹一下午壳子，从隔壁子王嬢嬢娃儿耍朋友，扯到美国大选，中间还掺几盘斗地主。他说我们四川人简直是把摸鱼刻进骨子里头咯！\n[S2] <|Sichuan|>你那个朋友说得倒是有点儿趣，但他莫看到精髓噻。摆龙门阵哪是摸鱼嘛，这是我们川渝人特有的交际方式，更是一种活法。外省人天天说的松弛感，根根儿就在这龙门阵里头。今天我们就要好生摆一哈，为啥子四川人活得这么舒坦。就先从茶馆这个老窝子说起，看它咋个成了我们四川人的魂儿！",
    ],
    [
        S1_PROMPT_WAV,
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
        "<|Yue|>真係冇讲错啊！攀山滑雪嘅语言专家几巴闭，都唔及我听日拖成副身家去景德镇玩泥巴，呢铺真系发哂白日梦咯！",
        S2_PROMPT_WAV,
        "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
        "<|Yue|>咪搞错啊！陕北民谣响度唱咗几十年，黑神话边有咁大面啊？你估佢哋抄游戏咩！",
        "[S1] <|Yue|>哈囉大家好啊，歡迎收聽我哋嘅節目。喂，我今日想問你樣嘢啊，你覺唔覺得，嗯，而家揸電動車，最煩，最煩嘅一樣嘢係咩啊？\n[S2] <|Yue|>梗係充電啦。大佬啊，搵個位都已經好煩，搵到個位仲要喺度等，你話快極都要半個鐘一個鐘，真係，有時諗起都覺得好冇癮。\n[S1] <|Yue|>係咪先。如果我而家同你講，充電可以快到同入油差唔多時間，你信唔信先？喂你平時喺油站入滿一缸油，要幾耐啊？五六分鐘？\n[S2] <|Yue|>差唔多啦，七八分鐘，點都走得啦。電車喎，可以做到咁快？你咪玩啦。",
    ],
    [
        S1_PROMPT_WAV,
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
        "<|Henan|>俺这不是怕恁路上不得劲儿嘛！那景德镇瓷泥可娇贵着哩，得先拿咱河南人这实诚劲儿给它揉透喽。",
        S2_PROMPT_WAV,
        "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
        "<|Henan|>恁这想法真闹挺！陕北民谣比黑神话早几百年都有了，咱可不兴这弄颠倒啊，中不？恁这想法真闹挺！那陕北民谣在黄土高坡响了几百年，咋能说是跟黑神话学的咧？咱得把这事儿捋直喽，中不中！",
        "[S1] <|Henan|>哎，大家好啊，欢迎收听咱这一期嘞《瞎聊呗，就这么说》，我是恁嘞老朋友，燕子。\n[S2] <|Henan|>大家好，我是老张。燕子啊，今儿瞅瞅你这个劲儿，咋着，是有啥可得劲嘞事儿想跟咱唠唠？\n[S1] <|Henan|>哎哟，老张，你咋恁懂我嘞！我跟你说啊，最近我刷手机，老是刷住些可逗嘞方言视频，特别是咱河南话，咦～我哩个乖乖，一听我都憋不住笑，咋说嘞，得劲儿哩很，跟回到家一样。\n[S2] <|Henan|>你这回可算说到根儿上了！河南话，咱往大处说说，中原官话，它真嘞是有一股劲儿搁里头。它可不光是说话，它脊梁骨后头藏嘞，是咱一整套、鲜鲜活活嘞过法儿，一种活人嘞道理。\n[S1] <|Henan|>活人嘞道理？哎，这你这一说，我嘞兴致“腾”一下就上来啦！觉住咱这嗑儿，一下儿从搞笑视频蹿到文化顶上了。那你赶紧给我白话白话，这里头到底有啥道道儿？我特别想知道——为啥一提起咱河南人，好些人脑子里“蹦”出来嘞头一个词儿，就是实在？这个实在，骨子里到底是啥嘞？",
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
    # Dialogue input textbox
    dialogue_text_input_label=dict(
        en="Dialogue Text Input",
        zh="合成文本输入",
    ),
    dialogue_text_input_placeholder=dict(
        en="[S1]text[S2]text[S1]text...",
        zh="[S1]文本[S2]文本[S1]文本...",
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
        en="Please provide prompt audio and text for both speaker 1 and speaker 2",
        zh="请提供说话人 1 与说话人 2 的参考语音与参考文本",
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

def check_dialogue_text(text_list: List[str]) -> bool:
    if len(text_list) == 0:
        return False
    for text in text_list:
        if not (
            check_monologue_text(text, "[S1]")
            or check_monologue_text(text, "[S2]")
            or check_monologue_text(text, "[S3]")
            or check_monologue_text(text, "[S4]")
        ):
            return False
    return True

def process_single(target_text_list, prompt_wav_list, prompt_text_list, use_dialect_prompt, dialect_prompt_text):
    spks, texts = [], []
    for target_text in target_text_list:
        pattern = r'(\[S[1-9]\])(.+)'
        match = re.match(pattern, target_text)
        text, spk = match.group(2), int(match.group(1)[2])-1
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
    spk1_prompt_text: str | None = "",
    spk1_prompt_audio: str | None = None,
    spk1_dialect_prompt_text: str | None = "",
    spk2_prompt_text: str | None = "",
    spk2_prompt_audio: str | None = None,
    spk2_dialect_prompt_text: str | None = "",
    seed: int = 1988,
):
    
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Check prompt info
    target_text_list: List[str] = re.findall(r"(\[S[0-9]\][^\[\]]*)", target_text)
    target_text_list = [text.strip() for text in target_text_list]
    if not check_dialogue_text(target_text_list):
        gr.Warning(message=i18n("warn_invalid_dialogue_text"))
        return None

    # Go synthesis
    progress_bar = gr.Progress(track_tqdm=True)
    prompt_wav_list = [spk1_prompt_audio, spk2_prompt_audio]
    prompt_text_list = [spk1_prompt_text, spk2_prompt_text] 
    use_dialect_prompt = spk1_dialect_prompt_text.strip()!="" or spk2_dialect_prompt_text.strip()!=""
    dialect_prompt_text_list = [spk1_dialect_prompt_text, spk2_dialect_prompt_text]
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

        with gr.Row():

            with gr.Column(scale=1):
                with gr.Group(visible=True) as spk1_prompt_group:
                    spk1_prompt_audio = gr.Audio(
                        label=i18n("spk1_prompt_audio_label"),
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )
                    spk1_prompt_text = gr.Textbox(
                        label=i18n("spk1_prompt_text_label"),
                        placeholder=i18n("spk1_prompt_text_placeholder"),
                        lines=3,
                    )
                    spk1_dialect_prompt_text = gr.Textbox(
                        label=i18n("spk1_dialect_prompt_text_label"),
                        placeholder=i18n("spk1_dialect_prompt_text_placeholder"),
                        value="",
                        lines=3,
                    )

            with gr.Column(scale=1, visible=True):
                with gr.Group(visible=True) as spk2_prompt_group:
                    spk2_prompt_audio = gr.Audio(
                        label=i18n("spk2_prompt_audio_label"),
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )
                    spk2_prompt_text = gr.Textbox(
                        label=i18n("spk2_prompt_text_label"),
                        placeholder=i18n("spk2_prompt_text_placeholder"),
                        lines=3,
                    )
                    spk2_dialect_prompt_text = gr.Textbox(
                        label=i18n("spk2_dialect_prompt_text_label"),
                        placeholder=i18n("spk2_dialect_prompt_text_placeholder"),
                        value="",
                        lines=3,
                    )

            with gr.Column(scale=2):
                with gr.Row():
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


        with gr.Row():
            inputs_for_examples = [
                spk1_prompt_audio,
                spk1_prompt_text,
                spk1_dialect_prompt_text,
                spk2_prompt_audio,
                spk2_prompt_text,
                spk2_dialect_prompt_text,
                dialogue_text_input,
            ]
            
            gr.Examples(
                examples=EXAMPLES_LIST,
                inputs=inputs_for_examples,
                label="播客模板示例 (点击加载)",
                examples_per_page=5,
            )
        
        with gr.Accordion("方言提示文本 (Dialect Prompt) 选择器", open=False):
            gr.Markdown("选择方言后，请分别为 S1 和 S2 选择一个示例。")
            dialect_selector = gr.Dropdown(
                label="选择方言 (Select Dialect)", 
                choices=DIALECT_CHOICES, 
                value="(无)",
                interactive=True
            )
            with gr.Row():
                s1_dialect_example_selector = gr.Dropdown(
                    label="S1 方言示例 (S1 Dialect Example)", 
                    choices=["(请先选择方言)"], 
                    value="(无)",
                    interactive=True,
                    elem_classes="gradio-dropdown" 
                )
                s2_dialect_example_selector = gr.Dropdown(
                    label="S2 方言示例 (S2 Dialect Example)", 
                    choices=["(请先选择方言)"], 
                    value="(无)",
                    interactive=True,
                    elem_classes="gradio-dropdown" 
                )
        
        dialect_selector.change(
            fn=update_example_choices,
            inputs=[dialect_selector],
            outputs=[s1_dialect_example_selector, s2_dialect_example_selector]
        )
        
        s1_dialect_example_selector.change(
            fn=update_prompt_text,
            inputs=[dialect_selector, s1_dialect_example_selector],
            outputs=[spk1_dialect_prompt_text]
        )
        
        s2_dialect_example_selector.change(
            fn=update_prompt_text,
            inputs=[dialect_selector, s2_dialect_example_selector],
            outputs=[spk2_dialect_prompt_text]
        )

        def _change_component_language(lang):
            global global_lang
            global_lang = ["zh", "en"][lang]
            return [
                
                # spk1_prompt_{audio,text,dialect_prompt_text}
                gr.update(label=i18n("spk1_prompt_audio_label")),
                gr.update(
                    label=i18n("spk1_prompt_text_label"),
                    placeholder=i18n("spk1_prompt_text_placeholder"),
                ),
                gr.update(
                    label=i18n("spk1_dialect_prompt_text_label"),
                    placeholder=i18n("spk1_dialect_prompt_text_placeholder"),
                ),
                # spk2_prompt_{audio,text}
                gr.update(label=i18n("spk2_prompt_audio_label")),
                gr.update(
                    label=i18n("spk2_prompt_text_label"),
                    placeholder=i18n("spk2_prompt_text_placeholder"),
                ),
                gr.update(
                    label=i18n("spk2_dialect_prompt_text_label"),
                    placeholder=i18n("spk2_dialect_prompt_text_placeholder"),
                ),
                # dialogue_text_input
                gr.update(
                    label=i18n("dialogue_text_input_label"),
                    placeholder=i18n("dialogue_text_input_placeholder"),
                ),
                # generate_btn
                gr.update(value=i18n("generate_btn_label")),
                # generate_audio
                gr.update(label=i18n("generated_audio_label")),
            ]

        lang_choice.change(
            fn=_change_component_language,
            inputs=[lang_choice],
            outputs=[
                spk1_prompt_audio,
                spk1_prompt_text,
                spk1_dialect_prompt_text,
                spk2_prompt_audio,
                spk2_prompt_text,
                spk2_dialect_prompt_text,
                dialogue_text_input,
                generate_btn,
                generate_audio,
            ],
        )
        
        generate_btn.click(
            fn=dialogue_synthesis_function,
            inputs=[
                dialogue_text_input,
                spk1_prompt_text,
                spk1_prompt_audio,
                spk1_dialect_prompt_text,
                spk2_prompt_text,
                spk2_prompt_audio,
                spk2_dialect_prompt_text,
                seed_input,
            ],
            outputs=[generate_audio],
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
    parser.add_argument('--port',
                        type=int,
                        default=7860,
                        help='gradio port for web app')
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
    page.launch(share=False, server_name="0.0.0.0", server_port=args.port)
