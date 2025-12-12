#!/bin/bash
# =============================
# SoulX-Podcast 启动脚本
# =============================

# 初始化 conda 环境
source /root/anaconda3/etc/profile.d/conda.sh

# 激活环境
conda activate soulxpodcast

# 进入项目目录
cd /ygq/rag/workspace/my-Soul-Podcast/SoulX-Podcast

# 启动 web 界面
python3 webui.py --model_path ../../SoulX-Podcast-main/pretrained_models/SoulX-Podcast-1.7B-dialect
