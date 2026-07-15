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
# 如果遇到 CUDA OOM 错误，可以降低 --gpu_memory_utilization 参数（例如 0.3 或 0.4）
python3 webui.py --gpu_memory_utilization 0.3 --llm_engine vllm --model_path /ygq/rag/workspace/SoulX-Podcast-main/pretrained_models/SoulX-Podcast-1.7B-dialect 
# python3 webui.py  --model_path /ygq/rag/workspace/SoulX-Podcast-main/pretrained_models/SoulX-Podcast-1.7B-dialect 
