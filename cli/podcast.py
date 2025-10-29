import os
import json
import torch
import argparse

import s3tokenizer
import soundfile as sf

from soulxpodcast.config import SamplingParams
from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.utils.infer_utils import initiate_model, process_single_input


def run_inference(
    inputs: dict,
    model_path: str,
    output_path: str,
    llm_engine: str = "hf",
    fp16_flow: bool = False,
    seed: int = 1988,
):
    
    model, dataset = initiate_model(seed, model_path, llm_engine, fp16_flow)
    
    data = process_single_input(
        dataset,
        inputs['text'],
        inputs['prompt_wav'],
        inputs['prompt_text'],
        inputs['use_dialect_prompt'],
        inputs['dialect_prompt_text'],
    )

    print("[INFO] Start inference...")
    results_dict = model.forward_longform(**data)

    target_audio = None
    for wav in results_dict["generated_wavs"]:
        if target_audio is None:
            target_audio = wav
        else:
            target_audio = torch.cat([target_audio, wav], dim=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, target_audio.cpu().squeeze(0).numpy(), 24000)
    print(f"[INFO] Saved synthesized audio to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to the input JSON file")
    parser.add_argument("--model_path", required=True, help="Path to the model file")
    parser.add_argument("--output_path", default="outputs/result.wav", help="Path to the output audio file")
    parser.add_argument("--llm_engine", default="hf", choices=["hf", "vllm"], help="Inference engine to use")
    parser.add_argument("--fp16_flow", action="store_true", help="Enable FP16 flow")
    parser.add_argument("--seed", type=int, default=1988, help="Random seed")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    inputs = podcast_format_parser(data)
    run_inference(
        inputs=inputs,
        model_path=args.model_path,
        output_path=args.output_path,
        llm_engine=args.llm_engine,
        fp16_flow=args.fp16_flow,
        seed=args.seed,
    )
