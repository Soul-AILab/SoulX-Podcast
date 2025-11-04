<div align="center">
    <h1>
    SoulX-Podcast
    </h1>
    <p>
    Official inference code for <br>
    <b><em>SoulX-Podcast: Towards Realistic Long-form Podcasts with Dialectal and Paralinguistic Diversity</em></b>
    </p>
    <p>
    <!-- <img src="assets/XiaoHongShu_Logo.png" alt="Institution 4" style="width: 102px; height: 48px;"> -->
    <img src="assets/SoulX-Podcast-log.jpg" alt="SoulX-Podcast_Logo" style="width: 200px; height: 68px;">
    </p>
    <p>
    </p>
    <a href="https://soul-ailab.github.io/soulx-podcast/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="version"></a>
    <a href="https://huggingface.co/collections/Soul-AILab/soulx-podcast"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF-model"></a>
    <a href="https://arxiv.org/pdf/2510.23541"><img src='https://img.shields.io/badge/Report-Github?label=Technical&color=red' alt="technical report"></a>
    <a href="https://huggingface.co/Soul-AILab/spaces"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue' alt="HF-demo"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Podcast"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache-2.0"></a>
</div>


<p align="center">
   <h1>SoulX-Podcast: Towards Realistic Long-form Podcasts with Dialectal and Paralinguistic Diversity</h1>
<p>

##  Overview
SoulX-Podcast is designed for podcast-style multi-turn, multi-speaker dialogic speech generation, while also achieving superior performance in the conventional monologue TTS task.

To meet the higher naturalness demands of multi-turn spoken dialogue, SoulX-Podcast integrates a range of paralinguistic controls and supports both Mandarin and English, as well as several Chinese dialects, including Sichuanese, Henanese, and Cantonese, enabling more personalized podcast-style speech generation.


## Key Features 🔥

- **Long-form, multi-turn, multi-speaker dialogic speech generation**: SoulX-Podcast excels in generating high-quality, natural-sounding dialogic speech for multi-turn, multi-speaker scenarios.

- **Cross-dialectal, zero-shot voice cloning**: SoulX-Podcast supports zero-shot voice cloning across different Chinese dialects, enabling the generation of high-quality, personalized speech in any of the supported dialects.

- **Paralinguistic controls**: SoulX-Podcast supports a variety of paralinguistic events, such as ***laughter***, ***breathing*** and ***coughing*** to enhance the realism of synthesized results. These are represented as inline non-verbal tokens (special tokens) that you can insert into input text to request short non-speech events during generation.

### Non-verbal / Paralinguistic tags

The model accepts a set of non-verbal tokens which can be inserted directly into the prompt text to produce short paralinguistic events. Example tokens include:

- `<|laughter|>`
- `<|breathing|>`
- `<|coughing|>`

Example usage (manual text or inside your podcast JSON):

```
[S1]Hello everyone, welcome to the show. <|laughter|> Today we have a great lineup.<|breathing|> Let's get started. <|coughing|>


```

When included in the input, the model will attempt to synthesize short, natural-sounding non-verbal events at those positions. Use these sparingly—paralinguistic events are best when used to punctuate dialogue or emphasize conversational turns.

<table align="center">
  <tr>
    <td align="center"><br><img src="assets/performance_radar.png" width="80%" /></td>
  </tr>
</table>


## Demo Examples

**Zero-Shot Podcast Generation**
<div align="center">

<https://github.com/user-attachments/assets/a9d3da2a-aaff-49d0-a3c7-2bd3c0b6d5eb>

</div>


**Cross-Dialectal Zero-Shot Podcast Generation**

🎙️ All prompt audio samples used in the following generations are in Mandarin.

🎙️ 以下音频生成采用的参考音频全部为普通话。

<div align="center">

<https://github.com/user-attachments/assets/982d799b-9f91-40a3-ab64-9e165166f788>

</div>
<div align="center">

<https://github.com/user-attachments/assets/d0a59d7b-27c9-4b47-8242-f7630814c1e9>

</div>

<div align="center">

<https://github.com/user-attachments/assets/a53ff35c-1e2b-42d9-9ef4-279164574646>

</div>

For more examples, see [demo page](https://soul-ailab.github.io/soulx-podcast/).

## Gradio UI (Interactive demo)

A lightweight Gradio-based UI is included to help you run quick experiments locally. The UI exposes two modes:

- Single Speaker: one-off utterance synthesis with an optional reference audio/text.
- Dual Speaker: multi-turn dialogue synthesis. You can provide a multi-line dialogue using either the bracketed tags `[S1]` / `[S2]` or the prefix form `S1:` / `S2:`. Example:

```
[S1] Hello everyone, welcome to the show. <|laughter|>
[S2] Thanks — happy to be here. <|breathing|>
```

The Dual Speaker tab accepts up to two reference audios and optional reference texts (one per speaker). If your dialogue references a speaker (for example `[S2]`) but you only supplied one reference, the UI will create a short silent fallback audio for the missing speaker so the preprocessing pipeline has a valid entry.

Files:

- `gradio_app.py` — the Gradio app entrypoint (run with `python gradio_app.py`).
- `example/gradio/README.md` — usage notes and examples for the Gradio UI.
- `example/gradio/smoke_test.py` — lightweight smoke test that validates input handling and can optionally run a full inference with `--run`.

Quick start (from project root):

```bash
pip install -r requirements.txt
python gradio_app.py
# open http://localhost:7860 in your browser
```


## 🚀 News
- **[2025-10-31]** Deploy an online demo on [Hugging Face Spaces](https://huggingface.co/Soul-AILab/spaces).

- **[2025-10-30]** Add example scripts for monologue TTS and support a WebUI for easy inference.

- **[2025-10-29]** We are excited to announce that the latest SoulX-Podcast checkpoint is now available on Hugging Face! You can access it directly from [SoulX-Podcast-hugging-face](https://huggingface.co/collections/Soul-AILab/soulx-podcast).

- **[2025-10-28]** Our paper on this project has been published! You can read it here: [SoulX-Podcast](https://arxiv.org/pdf/2510.23541).

## Install

### Clone and Install
Here are instructions for installing on Linux.
- Clone the repo
```
git clone git@github.com:Soul-AILab/SoulX-Podcast.git
cd SoulX-Podcast
```
- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:
```
conda create -n soulxpodcast -y python=3.11
conda activate soulxpodcast
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```
# [Optional] VLLM accleration(Modified version from vllm 0.10.1)
```
cd runtime/vllm
docker build -t soulxpodcast:v1.0 .
# Mounts the host directory at LOCAL_RESOURCE_PATH to CONTAINER_RESOURCE_PATH in the container, enabling file sharing between the host system and container. To access the web application, add -p LOCAL_PORT:CONTAINER_PORT
# example: docker run -it --runtime=nvidia  --name soulxpodcast  -v /mnt/data:/mnt/data -p 7860:7860 soulxpodcast:v1.0
docker run -it --runtime=nvidia  --name soulxpodcast  -v LOCAL_RESOURCE_PATH:CONTAINER_RESOURCE_PATH soulxpodcast:v1.0
```

### Model Download

```sh
pip install -U huggingface_hub

# base model
huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B --local-dir pretrained_models/SoulX-Podcast-1.7B

# dialectal model
huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B-dialect --local-dir pretrained_models/SoulX-Podcast-1.7B-dialect
```


Download via python:
```python
from huggingface_hub import snapshot_download

# base model
snapshot_download("Soul-AILab/SoulX-Podcast-1.7B", local_dir="pretrained_models/SoulX-Podcast-1.7B") 

# dialectal model
snapshot_download("Soul-AILab/SoulX-Podcast-1.7B-dialect", local_dir="pretrained_models/SoulX-Podcast-1.7B-dialect") 

```

Download via git clone:
```sh
mkdir -p pretrained_models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# base model
git clone https://huggingface.co/Soul-AILab/SoulX-Podcast-1.7B pretrained_models/SoulX-Podcast-1.7B

# dialectal model
git clone https://huggingface.co/Soul-AILab/SoulX-Podcast-1.7B-dialect pretrained_models/SoulX-Podcast-1.7B-dialect
```


### Basic Usage

You can simply run the demo with the following commands:
``` sh
# dialectal inference
bash example/infer_dialogue.sh
```

### WebUI

You can simply run the webui with the following commands:
``` sh
# Base Model:
python3 webui.py --model_path pretrained_models/SoulX-Podcast-1.7B

# If you want to experience dialect podcast generation, use the dialectal model:
python3 webui.py --model_path pretrained_models/SoulX-Podcast-1.7B-dialect


```


## TODOS
- [x] Add example scripts for monologue TTS.
- [x] Publish the [technical report](https://arxiv.org/pdf/2510.23541).
- [x] Develop a WebUI for easy inference.
- [x] Deploy an online demo on [Hugging Face Spaces](https://huggingface.co/Soul-AILab/spaces).
- [x] Dockerize the project with vLLM support.
- [ ] Add support for streaming inference.

## Citation

```bibtex
@misc{SoulXPodcast,
  title        = {SoulX-Podcast: Towards Realistic Long-form Podcasts with Dialectal and Paralinguistic Diversity},
  author       = {Hanke Xie and Haopeng Lin and Wenxiao Cao and Dake Guo and Wenjie Tian and Jun Wu and Hanlin Wen and Ruixuan Shang and Hongmei Liu and Zhiqi Jiang and Yuepeng Jiang and Wenxi Chen and Ruiqi Yan and Jiale Qian and Yichao Yan and Shunshun Yin and Ming Tao and Xie Chen and Lei Xie and Xinsheng Wang},
  year         = {2025},
  archivePrefix={arXiv},
  url          = {https://arxiv.org/abs/2510.23541}
}

```

## License

We use the Apache 2.0 license. Researchers and developers are free to use the codes and model weights of our SoulX-Podcast. Check the license at [LICENSE](LICENSE) for more details.


## Acknowledge
- This repo benefits from [FlashCosyVoice](https://github.com/xingchensong/FlashCosyVoice/tree/main)


##  Usage Disclaimer
This project provides a speech synthesis model for podcast generation capable of zero-shot voice cloning, intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:

Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities.

Ensure compliance with local laws and regulations when using this model and uphold ethical standards.

The developers assume no liability for any misuse of this model.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.

## Contact Us
If you are interested in leaving a message to our work, feel free to email hkxie@mail.nwpu.edu.cn or linhaopeng@soulapp.cn or lxie@nwpu.edu.cn or wangxinsheng@soulapp.cn

You’re welcome to join our WeChat group for technical discussions, updates.
<p align="center">
  <!-- <em>Due to group limits, if you can't scan the QR code, please add my WeChat for group access  -->
      <!-- : <strong>Tiamo James</strong></em> -->
  <br>
  <span style="display: inline-block; margin-right: 10px;">
    <img src="assets/wechat2.jpg" width="300" alt="WeChat Group QR Code"/>
  </span>
  <!-- <span style="display: inline-block;">
    <img src="assets/wechat_tiamo.jpg" width="300" alt="WeChat QR Code"/>
  </span> -->
</p>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Soul-AILab/SoulX-Podcast&type=date&legend=top-left)](https://www.star-history.com/#Soul-AILab/SoulX-Podcast&type=date&legend=top-left)
<div align="center">
    <h1>
    SoulX-Podcast
    </h1>
    <p>
    Official inference code for <br>
    <b><em>SoulX-Podcast: Towards Realistic Long-form Podcasts with Dialectal and Paralinguistic Diversity</em></b>
    </p>
    <p>
    <!-- <img src="assets/XiaoHongShu_Logo.png" alt="Institution 4" style="width: 102px; height: 48px;"> -->
    <img src="assets/SoulX-Podcast-log.jpg" alt="SoulX-Podcast_Logo" style="width: 200px; height: 68px;">
    </p>
    <p>
    </p>
    <a href="https://soul-ailab.github.io/soulx-podcast/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="version"></a>
    <a href="https://huggingface.co/collections/Soul-AILab/soulx-podcast"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF-model"></a>
    <a href="https://arxiv.org/pdf/2510.23541"><img src='https://img.shields.io/badge/Report-Github?label=Technical&color=red' alt="technical report"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Podcast"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache-2.0"></a>
</div>


<p align="center">
   <h1>SoulX-Podcast: Towards Realistic Long-form Podcasts with Dialectal and Paralinguistic Diversity</h1>
<p>

##  Overview
SoulX-Podcast is designed for podcast-style multi-turn, multi-speaker dialogic speech generation, while also achieving superior performance in the conventional monologue TTS task.

To meet the higher naturalness demands of multi-turn spoken dialogue, SoulX-Podcast integrates a range of paralinguistic controls and supports both Mandarin and English, as well as several Chinese dialects, including Sichuanese, Henanese, and Cantonese, enabling more personalized podcast-style speech generation.


## Key Features 🔥

- **Long-form, multi-turn, multi-speaker dialogic speech generation**: SoulX-Podcast excels in generating high-quality, natural-sounding dialogic speech for multi-turn, multi-speaker scenarios.

- **Cross-dialectal, zero-shot voice cloning**: SoulX-Podcast supports zero-shot voice cloning across different Chinese dialects, enabling the generation of high-quality, personalized speech in any of the supported dialects.

- **Paralinguistic controls**: SoulX-Podcast supports a variety of paralinguistic events, as as ***laugher*** and ***sighs*** to enhance the realism of synthesized results.

- **Paralinguistic controls**: SoulX-Podcast supports a variety of paralinguistic events, such as ***laughter***, ***breathing*** and ***coughing*** to enhance the realism of synthesized results. These are represented as inline non-verbal tokens (special tokens) that you can insert into input text to request short non-speech events during generation.

### Non-verbal / Paralinguistic tags

The model accepts a set of non-verbal tokens which can be inserted directly into the prompt text to produce short paralinguistic events. Example tokens include:

- `<|laughter|>`
- `<|breathing|>`
- `<|coughing|>`

Example usage (manual text or inside your podcast JSON):

```
[S1]Hello everyone, welcome to the show. <|laughter|> Today we have a great lineup.<|breathing|> Let's get started. <|coughing|>


```

When included in the input, the model will attempt to synthesize short, natural-sounding non-verbal events at those positions. Use these sparingly—paralinguistic events are best when used to punctuate dialogue or emphasize conversational turns.

<table align="center">
  <tr>
    <td align="center"><br><img src="assets/performance_radar.png" width="80%" /></td>
  </tr>
</table>


## Demo Examples

**Zero-Shot Podcast Generation**
<div align="center">

<https://github.com/user-attachments/assets/a9d3da2a-aaff-49d0-a3c7-2bd3c0b6d5eb>

</div>


**Cross-Dialectal Zero-Shot Podcast Generation**

🎙️ All prompt audio samples used in the following generations are in Mandarin.

🎙️ 以下音频生成采用的参考音频全部为普通话。

<div align="center">

<https://github.com/user-attachments/assets/982d799b-9f91-40a3-ab64-9e165166f788>

</div>
<div align="center">

<https://github.com/user-attachments/assets/d0a59d7b-27c9-4b47-8242-f7630814c1e9>

</div>

<div align="center">

<https://github.com/user-attachments/assets/a53ff35c-1e2b-42d9-9ef4-279164574646>

</div>

For more examples, see [demo page](https://soul-ailab.github.io/soulx-podcast/).

## Gradio UI (Interactive demo)

A lightweight Gradio-based UI is included to help you run quick experiments locally. The UI exposes two modes:

- Single Speaker: one-off utterance synthesis with an optional reference audio/text.
- Dual Speaker: multi-turn dialogue synthesis. You can provide a multi-line dialogue using either the bracketed tags `[S1]` / `[S2]` or the prefix form `S1:` / `S2:`. Example:

```
[S1] Hello everyone, welcome to the show. <|laughter|>
[S2] Thanks — happy to be here. <|breathing|>
```

The Dual Speaker tab accepts up to two reference audios and optional reference texts (one per speaker). If your dialogue references a speaker (for example `[S2]`) but you only supplied one reference, the UI will create a short silent fallback audio for the missing speaker so the preprocessing pipeline has a valid entry.

Files:

- `gradio_app.py` — the Gradio app entrypoint (run with `python gradio_app.py`).
- `example/gradio/README.md` — usage notes and examples for the Gradio UI.
- `example/gradio/smoke_test.py` — lightweight smoke test that validates input handling and can optionally run a full inference with `--run`.

Quick start (from project root):

```bash
pip install -r requirements.txt
python gradio_app.py
# open http://localhost:7860 in your browser
```


## 🚀 News

- **[2025-10-29]** We are excited to announce that the latest SoulX-Podcast checkpoint is now available on Hugging Face! You can access it directly from [SoulX-Podcast-hugging-face](https://huggingface.co/collections/Soul-AILab/soulx-podcast).

- **[2025-10-28]** Our paper on this project has been published! You can read it here: [SoulX-Podcast](https://arxiv.org/pdf/2510.23541).

## Install

### Clone and Install
Here are instructions for installing on Linux.
- Clone the repo
```
git clone git@github.com:Soul-AILab/SoulX-Podcast.git
cd SoulX-Podcast
```
- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:
```
conda create -n soulxpodcast -y python=3.11
conda activate soulxpodcast
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

### Model Download

```sh
pip install -U huggingface_hub

# base model
huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B --local-dir pretrained_models/SoulX-Podcast-1.7B

# dialectal model
huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B-dialect --local-dir pretrained_models/SoulX-Podcast-1.7B-dialect
```


Download via python:
```python
from huggingface_hub import snapshot_download

# base model
snapshot_download("Soul-AILab/SoulX-Podcast-1.7B", local_dir="pretrained_models/SoulX-Podcast-1.7B") 

# dialectal model
snapshot_download("Soul-AILab/SoulX-Podcast-1.7B-dialect", local_dir="pretrained_models/SoulX-Podcast-1.7B-dialect") 

```

Download via git clone:
```sh
mkdir -p pretrained_models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# base model
git clone https://huggingface.co/Soul-AILab/SoulX-Podcast-1.7B pretrained_models/SoulX-Podcast-1.7B

# dialectal model
git clone https://huggingface.co/Soul-AILab/SoulX-Podcast-1.7B-dialect pretrained_models/SoulX-Podcast-1.7B-dialect
```


### Basic Usage

You can simply run the demo with the following commands:
``` sh
# dialectal inference
bash example/infer_dialogue.sh
```

## TODOs
- [ ] Add example scripts for monologue TTS.
- [x] Publish the [technical report](https://arxiv.org/pdf/2510.23541).
- [ ] Develop a WebUI for easy inference.
- [ ] Deploy an online demo on Hugging Face Spaces.
- [ ] Dockerize the project with vLLM support.
- [ ] Add support for streaming inference.

## Citation

```bibtex
@misc{SoulXPodcast,
  title        = {SoulX-Podcast: Towards Realistic Long-form Podcasts with Dialectal and Paralinguistic Diversity},
  author       = {Hanke Xie and Haopeng Lin and Wenxiao Cao and Dake Guo and Wenjie Tian and Jun Wu and Hanlin Wen and Ruixuan Shang and Hongmei Liu and Zhiqi Jiang and Yuepeng Jiang and Wenxi Chen and Ruiqi Yan and Jiale Qian and Yichao Yan and Shunshun Yin and Ming Tao and Xie Chen and Lei Xie and Xinsheng Wang},
  year         = {2025},
  archivePrefix={arXiv},
  url          = {https://arxiv.org/abs/2510.23541}
}

```

## License

We use the Apache 2.0 license. Researchers and developers are free to use the codes and model weights of our SoulX-Podcast. Check the license at [LICENSE](LICENSE) for more details.


## Acknowledge
- This repo benefits from [FlashCosyVoice](https://github.com/xingchensong/FlashCosyVoice/tree/main)


##  Usage Disclaimer
This project provides a speech synthesis model for podcast generation capable of zero-shot voice cloning, intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:

Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities.

Ensure compliance with local laws and regulations when using this model and uphold ethical standards.

The developers assume no liability for any misuse of this model.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.

## Contact us
If you are interested in leaving a message to our work, feel free to email hkxie@mail.nwpu.edu.cn or linhaopeng@soulapp.cn or lxie@nwpu.edu.cn or wangxinsheng@soulapp.cn

You’re welcome to join our WeChat group for technical discussions, updates.
<p align="center">
  <em>Due to group limits, if you can't scan the QR code, please add my WeChat for group access 
      <!-- : <strong>Tiamo James</strong></em> -->
  <br>
  <span style="display: inline-block; margin-right: 10px;">
    <img src="assets/wechat.jpg" width="300" alt="WeChat Group QR Code"/>
  </span>
  <span style="display: inline-block;">
    <img src="assets/wechat_tiamo.jpg" width="300" alt="WeChat QR Code"/>
  </span>
</p>

<!-- <p align="center">
    <img src="src/figs/npu@aslp.jpeg" width="500"/>
</p -->
<!-- <img src="assets/wechat.jpg -->
>>>>>>> 38a37ce (WIP: save local changes before rebase)
