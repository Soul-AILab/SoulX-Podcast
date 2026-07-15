# -*- coding: utf-8 -*-
"""
SoulX-Podcast WebUI Entry Point

This file serves as the backward-compatible entry point for launching the WebUI.
The actual implementation has been modularized into the webui/ package.
"""

import importlib.util
from datetime import datetime
from argparse import ArgumentParser

import torch
import numpy as np
import random
from tqdm import tqdm

from soulxpodcast.config import Config, SoulXPodcastLLMConfig

from webui import render_interface, initiate_model


def get_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--model_path',
        required=True,
        type=str,
        help='model path'
    )
    parser.add_argument(
        '--llm_engine',
        type=str,
        default="hf",
        help='model execute engine'
    )
    parser.add_argument(
        '--fp16_flow',
        action='store_true',
        help='enable fp16 flow'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1988,
        help='random seed for generation'
    )
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.5,
        help='GPU memory utilization ratio for VLLM (default: 0.5, lower if OOM)'
    )
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=8192,
        help='Maximum model length for VLLM (default: 8192)'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Initiate model
    hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
        initial_values={"fp16_flow": args.fp16_flow},
        json_file=f"{args.model_path}/soulxpodcast_config.json"
    )
    
    llm_engine = args.llm_engine
    if llm_engine == "vllm":
        if not importlib.util.find_spec("vllm"):
            llm_engine = "hf"
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [WARNING]: No install VLLM, switch to hf engine.")
    
    config = Config(
        model=args.model_path,
        enforce_eager=True,
        llm_engine=llm_engine,
        hf_config=hf_config,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    initiate_model(config)
    print("[INFO] SoulX-Podcast loaded")
    
    page = render_interface()
    page.queue()
    page.launch(share=False)
