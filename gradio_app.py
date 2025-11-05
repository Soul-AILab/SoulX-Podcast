import os
import json
import tempfile
from typing import Optional

import gradio as gr
import torch
import soundfile as sf
import numpy as np
import torchaudio
import importlib
import re

# Use lower-level utilities so we can cache model and dataset in memory between runs
from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.utils.infer_utils import initiate_model, process_single_input

# Simple in-memory cache for model and dataset
MODEL_CACHE = {
    "model_path": None,
    "llm_engine": None,
    "fp16_flow": None,
    "seed": None,
    "model": None,
    "dataset": None,
}


def load_model_if_needed(model_path, llm_engine, fp16_flow, seed):
    """Load model and dataset into MODEL_CACHE if not loaded or if parameters changed.

    Returns (model, dataset, loaded_new)
    """
    # Normalize inputs to avoid reloads caused by equivalent but non-identical representations
    try:
        # Normalize path robustly: abspath -> realpath -> normpath -> normcase
        norm_model_path = os.path.normcase(os.path.normpath(os.path.realpath(os.path.abspath(model_path)))) if model_path else model_path
    except Exception:
        norm_model_path = model_path
    norm_llm_engine = str(llm_engine) if llm_engine is not None else llm_engine
    norm_fp16_flow = bool(fp16_flow)
    try:
        norm_seed = int(seed)
    except Exception:
        norm_seed = seed

    loaded_new = False
    if (
        MODEL_CACHE["model_path"] != norm_model_path
        or MODEL_CACHE["llm_engine"] != norm_llm_engine
        or MODEL_CACHE["fp16_flow"] != norm_fp16_flow
        or MODEL_CACHE["seed"] != norm_seed
        or MODEL_CACHE["model"] is None
        or MODEL_CACHE["dataset"] is None
    ):
        # free previous model references if present to avoid duplicate GPU allocations
        try:
            if MODEL_CACHE.get("model") is not None:
                # attempt to delete model and dataset and free CUDA memory before loading new one
                try:
                    del MODEL_CACHE["model"]
                except Exception:
                    pass
                try:
                    del MODEL_CACHE["dataset"]
                except Exception:
                    pass
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        except Exception:
            pass

        MODEL_CACHE["model_path"] = norm_model_path
        MODEL_CACHE["llm_engine"] = norm_llm_engine
        MODEL_CACHE["fp16_flow"] = norm_fp16_flow
        MODEL_CACHE["seed"] = norm_seed
        # log loading action
        try:
            from datetime import datetime

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading model from {norm_model_path} (engine={norm_llm_engine}, fp16_flow={norm_fp16_flow})")
        except Exception:
            pass

        model, dataset = initiate_model(norm_seed, norm_model_path, norm_llm_engine, norm_fp16_flow)
        MODEL_CACHE["model"] = model
        MODEL_CACHE["dataset"] = dataset
        loaded_new = True
    return MODEL_CACHE["model"], MODEL_CACHE["dataset"], loaded_new


def save_uploaded_file(uploaded) -> Optional[str]:
    if uploaded is None:
        return None
    # Handle common Gradio return types: filepath (str), dict with 'name'/'tmp_path', or file-like
    # If it's already a filepath, return it
    try:
        if isinstance(uploaded, str):
            return uploaded
    except Exception:
        pass

    # If Gradio returned a dict with paths
    try:
        if isinstance(uploaded, dict):
            return uploaded.get("name") or uploaded.get("tmp_path") or uploaded.get("file")
    except Exception:
        pass

    # If it's a file-like object, try to write it to a temp file
    try:
        filename = getattr(uploaded, "name", None) or "upload"
        _, ext = os.path.splitext(filename)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        # Some file-likes have a .read() method
        if hasattr(uploaded, "read"):
            data = uploaded.read()
            # If data is str, encode
            if isinstance(data, str):
                data = data.encode("utf-8")
            tmp.write(data)
            tmp.flush()
            tmp.close()
            return tmp.name
        # If uploaded is raw bytes
        if isinstance(uploaded, (bytes, bytearray)):
            tmp.write(uploaded)
            tmp.flush()
            tmp.close()
            return tmp.name
        # If we couldn't handle the uploaded file, cleanup and return None
        try:
            tmp.close()
        except Exception:
            pass
        return None
    except Exception:
        # Any error while saving uploaded file -> treat as no file
        try:
            tmp.close()
        except Exception:
            pass
        return None

def infer_from_ui(
    json_file,
    manual_text,
    prompt_wav_file,
    sample_choice_s1=None,
    use_sample_s1=False,
    prompt_text="",

    speaker2_prompt_wav_file=None,
    speaker2_prompt_text="",
    dialogue_text="",
    use_dialect_prompt=False,
    dialect_prompt_text="",
    model_choice=None,
    model_path=None,
    output_path="outputs/output.wav",
    llm_engine="hf",
    fp16_flow=True,
    seed=42,
    # optional dual-speaker params
    sample_choice_s2=None,
    use_sample_s2=False,
    auto_fill_s2=False,
):
    try:
        # gradio may return a filepath (str), a dict, or a file-like
        if isinstance(json_file, str):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(json_file, dict):
            # dict may contain 'name' or 'tmp_path'
            path = json_file.get("name") or json_file.get("tmp_path") or json_file.get("file")
            if path:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                # try to read file-like entries
                data = json.load(json_file)
        else:
            # file-like object
            data = json.load(json_file)
        inputs = podcast_format_parser(data)
        # If the JSON referred to sample identifiers (e.g. a repo filename or 'sample1'),
        # resolve those to actual files under example/audios when available so the UI
        # behaves as if the user selected the sample in the dropdown.
        try:
            sample_dir = os.path.join(os.path.dirname(__file__), "example", "audios")
            sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith('.wav')]
        except Exception:
            sample_files = []

        def _resolve_sample_entry(val):
            try:
                if not val:
                    return val
                # already an absolute path and exists
                if os.path.isabs(val) and os.path.exists(val):
                    return val
                # maybe it's a relative path from cwd
                cand = os.path.join(os.getcwd(), val)
                if os.path.exists(cand):
                    return cand
                # direct match to a repo filename
                if val in sample_files:
                    return os.path.join(sample_dir, val)
                # normalized forms: 'sample1', 's1', '1' -> map by index
                s = str(val).lower().replace(" ", "").replace("_", "")
                m = re.match(r"^s?(\d+)$", s)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(sample_files):
                        return os.path.join(sample_dir, sample_files[idx])
                # fallback: return original value
                return val
            except Exception:
                return val

        try:
            # Check for uploaded reference audios (these should take precedence over dropdown selections)
            uploaded_prompt1 = save_uploaded_file(prompt_wav_file)
            uploaded_prompt2 = save_uploaded_file(speaker2_prompt_wav_file)

            pw = inputs.get("prompt_wav", [])
            if isinstance(pw, list):
                resolved = []
                for x in pw:
                    lx = str(x).lower() if x is not None else ""
                    if lx in ("sample1", "s1", "sample_1"):
                        # prefer uploaded file for speaker1, then UI-selected sample, then repo fallback
                        if uploaded_prompt1:
                            resolved.append(uploaded_prompt1)
                        else:
                            choice = sample_choice_s1 if 'sample_choice_s1' in locals() else None
                            if not choice:
                                choice = sample_files[0] if sample_files else None
                            resolved.append(_resolve_sample_entry(choice))
                    elif lx in ("sample2", "s2", "sample_2"):
                        # prefer uploaded file for speaker2, then UI-selected sample, then repo fallback
                        if uploaded_prompt2:
                            resolved.append(uploaded_prompt2)
                        else:
                            choice = sample_choice_s2 if 'sample_choice_s2' in locals() else None
                            if not choice:
                                choice = sample_files[1] if len(sample_files) > 1 else (sample_files[0] if sample_files else None)
                            resolved.append(_resolve_sample_entry(choice))
                    else:
                        resolved.append(_resolve_sample_entry(x))
                inputs["prompt_wav"] = resolved
        except Exception:
            pass
    except Exception:
        # Build a minimal inputs dict from manual fields when JSON parsing not provided
        key = "ui"
        # Priority: uploaded file > selected sample (if enabled) > silent fallback
        prompt_wav_path = save_uploaded_file(prompt_wav_file)
        # resolve sample path for speaker1
        try:
            sample_path_s1 = os.path.join(os.path.dirname(__file__), "example", "audios", sample_choice_s1) if sample_choice_s1 else None
        except Exception:
            sample_path_s1 = None

        if not prompt_wav_path and bool(use_sample_s1) and sample_path_s1 and os.path.exists(sample_path_s1):
            prompt_wav_path = sample_path_s1

        # If still empty, create a short silent audio file as a safe fallback
        if not prompt_wav_path:
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                # create 0.5s silent audio at 16kHz
                sr = 16000
                duration_s = 0.5
                samples = int(sr * duration_s)
                # torchaudio.save expects tensor shape (channels, samples)
                tensor = torch.zeros((1, samples), dtype=torch.float32)
                torchaudio.save(tmp.name, tensor, sr)
                prompt_wav_path = tmp.name
            except Exception:
                prompt_wav_path = ""

        # helper to create a silent wav when needed
        def _make_silent_wav():
            try:
                tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sr2 = 16000
                dur2 = 0.5
                samples2 = int(sr2 * dur2)
                tensor2 = torch.zeros((1, samples2), dtype=torch.float32)
                torchaudio.save(tmp2.name, tensor2, sr2)
                return tmp2.name
            except Exception:
                return ""

        # Speaker 2 support: allow a second uploaded reference and prompt text
        speaker2_prompt_wav_path = save_uploaded_file(speaker2_prompt_wav_file)
        # resolve sample path for speaker2
        try:
            sample_path_s2 = os.path.join(os.path.dirname(__file__), "example", "audios", sample_choice_s2) if sample_choice_s2 else None
        except Exception:
            sample_path_s2 = None

        if not speaker2_prompt_wav_path and bool(use_sample_s2) and sample_path_s2 and os.path.exists(sample_path_s2):
            speaker2_prompt_wav_path = sample_path_s2

        # auto-fill speaker2 with speaker1 sample if requested
        if not speaker2_prompt_wav_path and bool(auto_fill_s2) and sample_path_s1 and os.path.exists(sample_path_s1):
            speaker2_prompt_wav_path = sample_path_s1

        if not speaker2_prompt_wav_path:
            speaker2_prompt_wav_path = ""

        prompt_texts = [prompt_text] if prompt_text else [""]
        if speaker2_prompt_text:
            prompt_texts = [prompt_texts[0], speaker2_prompt_text]

        dialect_texts = [dialect_prompt_text] if dialect_prompt_text else [""]

        # If dialogue_text is provided, parse multi-line turns e.g. "S1: Hello\nS2: Hi"
        text_list = []
        spk_list = []
        if dialogue_text and dialogue_text.strip():
            for line in dialogue_text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                # Accept two common formats: bracketed tags like [S1]Hello and prefix forms like S1: Hello
                m = re.match(r"^\[\s*S?([12])\s*\]\s*(.*)$", line, re.IGNORECASE)
                if not m:
                    m = re.match(r"^S?([12])[:\)]\s*(.*)$", line, re.IGNORECASE)

                if m:
                    spk_idx = int(m.group(1)) - 1
                    utt = m.group(2).strip()
                    # If the utterance is not already tagged, add canonical [S#] tag
                    if not utt.startswith("[S"):
                        utt = f"[S{spk_idx+1}]{utt}"
                    text_list.append(utt)
                    spk_list.append(spk_idx)
                else:
                    # default to speaker1 when no explicit tag is found
                    utt = line
                    if not utt.startswith("[S"):
                        utt = f"[S1]{utt}"
                    text_list.append(utt)
                    spk_list.append(0)
        else:
            # Ensure manual text includes a speaker tag expected by the inference pipeline (e.g., [S1]...)
            if manual_text:
                if manual_text.strip().startswith("[S"):
                    text_entry = manual_text.strip()
                else:
                    text_entry = f"[S1]{manual_text.strip()}"
            else:
                text_entry = "[S1]"

            text_list = [text_entry]
            spk_list = [0]

        # Build prompt_wav list: include speaker2 if provided
        prompt_wav = [prompt_wav_path]
        if speaker2_prompt_wav_path:
            prompt_wav = [prompt_wav_path, speaker2_prompt_wav_path]

        # Ensure prompt_wav and prompt_texts have entries for all speakers referenced in text_list
        try:
            max_spk = max(spk_list) if spk_list else 0
        except Exception:
            max_spk = 0

        # pad prompt_wav
        while len(prompt_wav) <= max_spk:
            prompt_wav.append(_make_silent_wav())

        # pad prompt_texts
        while len(prompt_texts) <= max_spk:
            prompt_texts.append("")

        inputs = {
            "key": key,
            "prompt_text": prompt_texts,
            "prompt_wav": prompt_wav,
            "text": text_list,
            "spk": spk_list,
            "wav": output_path,
            "use_dialect_prompt": bool(use_dialect_prompt),
            "dialect_prompt_text": dialect_texts,
        }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # If a model was selected from the dropdown, prefer that over manual model_path textbox
    try:
        if model_choice:
            candidate = os.path.join(os.path.dirname(__file__), "pretrained_models", model_choice)
            if os.path.exists(candidate):
                model_path = candidate
            else:
                # allow absolute/relative paths typed into the dropdown
                model_path = model_choice
    except Exception:
        pass

    try:
        # Load model and dataset into memory if needed
        model, dataset, loaded_new = load_model_if_needed(model_path, llm_engine, fp16_flow, seed)

        # Prepare processed data using the shared dataset
        processed = process_single_input(
            dataset,
            inputs["text"],
            inputs["prompt_wav"],
            inputs["prompt_text"],
            inputs.get("use_dialect_prompt", False),
            inputs.get("dialect_prompt_text", [""]),
        )

        # Run model generation
        results_dict = model.forward_longform(**processed)

        # Concatenate generated wavs into single tensor
        target_audio = None
        for wav in results_dict.get("generated_wavs", []):
            if target_audio is None:
                target_audio = wav
            else:
                target_audio = torch.cat([target_audio, wav], dim=1)

        if target_audio is None:
            return "Inference produced no audio", None

        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, target_audio.cpu().squeeze(0).numpy(), 24000)

        status_msg = "Done"
        if loaded_new:
            status_msg = f"Model loaded from {model_path}. {status_msg}"
        else:
            status_msg = f"Reused cached model ({MODEL_CACHE['model_path']}). {status_msg}"

    except Exception as e:
        return f"Inference failed: {e}", None, None

    # Return status, audio path and currently loaded model path for Gradio to display
    return status_msg, output_path, MODEL_CACHE.get("model_path")


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# SoulX-Podcast — Interactive Demo\nUse this UI to synthesize podcast-style speech. You can use the Single Speaker tab for one-off utterances, or Dual Speaker for back-and-forth dialogues. Upload JSON or use the fields in each tab.")

        with gr.Tabs():
            with gr.TabItem("Single Speaker"):
                with gr.Row():
                    with gr.Column():
                        json_file = gr.File(label="Upload podcast JSON (optional) — follows example/podcast_script schema")
                        manual_text = gr.Textbox(label="Manual utterance (used when no JSON) — speaker tag [S1] will be added if omitted", lines=4, placeholder="Hello, this is a test.")
                        # Sample selector (files under example/audios)
                        try:
                            sample_dir = os.path.join(os.path.dirname(__file__), "example", "audios")
                            sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith('.wav')]
                        except Exception:
                            sample_files = []
                        sample_choice = gr.Dropdown(
                            label="Choose a sample reference (from repo)",
                            choices=sample_files,
                            value=("female_mandarin.wav" if "female_mandarin.wav" in sample_files else (sample_files[0] if sample_files else None)),
                        )
                        # (Resolved sample path display removed per UI simplification request)
                        # update helper to compute sample path when needed
                        def _sample_path(choice):
                            try:
                                if not choice:
                                    return ""
                                p = os.path.join(os.path.dirname(__file__), "example", "audios", choice)
                                return p if os.path.exists(p) else p
                            except Exception:
                                return ""
                        use_sample = gr.Checkbox(label="Use selected sample as reference audio (if no upload)", value=True)
                        # Use a simple Audio component; avoid `source` kwarg for compatibility
                        prompt_wav = gr.Audio(label="Or upload your own reference audio (takes precedence)", type="filepath")
                        prompt_text = gr.Textbox(label="Reference text (optional) — short transcript describing voice/style")
                        use_dialect_prompt = gr.Checkbox(label="Enable dialect prompt (use dialect-specific model)", value=False)
                        dialect_prompt_text = gr.Textbox(label="Dialect prompt (e.g. <|Henan|>我今天很高兴)", placeholder="<|Henan|>...", lines=2)

                    with gr.Column():
                        # Model selector: list local subdirectories under pretrained_models
                        try:
                            model_root = os.path.join(os.path.dirname(__file__), "pretrained_models")
                            model_dirs = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
                        except Exception:
                            model_dirs = []

                        default_model = None
                        if "SoulX-Podcast-1.7B-dialect" in model_dirs:
                            default_model = "SoulX-Podcast-1.7B-dialect"
                        elif model_dirs:
                            default_model = model_dirs[0]

                        model_choice = gr.Dropdown(label="Select a local model (from pretrained_models)", choices=model_dirs, value=default_model)
                        model_path = gr.Textbox(label="Local model directory (override)", value=(os.path.join("pretrained_models", default_model) if default_model else "pretrained_models/SoulX-Podcast-1.7B-dialect"))
                        output_path = gr.Textbox(label="Output WAV file (will be overwritten)", value="outputs/output_single.wav")
                        # Detect whether vllm is installed; if not, don't offer it as a primary choice (selecting it will fall back to hf)
                        try:
                            vllm_available = importlib.util.find_spec("vllm") is not None
                        except Exception:
                            vllm_available = False

                        llm_choices = ["hf", "vllm"] if vllm_available else ["hf"]
                        llm_default = "hf"
                        llm_engine = gr.Dropdown(label="LM engine (hf = HuggingFace, vllm = vLLM)", choices=llm_choices, value=llm_default)
                        if not vllm_available:
                            gr.Markdown("**Note:** vLLM is not installed in this environment. Selecting vllm will automatically fall back to the `hf` engine. To enable vLLM install it in your environment: `pip install vllm`.")
                        fp16_flow = gr.Checkbox(label="Use FP16 for flow model (recommended on supported GPUs)", value=True)
                        seed = gr.Number(label="Random seed (int)", value=42, precision=0)
                        run_single_btn = gr.Button("Run Single-Speaker Inference")
                        status = gr.Textbox(label="Job status / messages", interactive=False)
                        out_audio = gr.Audio(label="Play or download generated audio", type="filepath")
                        model_status = gr.Textbox(label="Model status (loaded path)", interactive=False)

                        # Hidden/default state values for single-tab invocation (used to satisfy Gradio input list)
                        speaker2_prompt_wav_state = gr.State(value=None)
                        speaker2_prompt_text_state = gr.State(value="")
                        dialogue_text_state = gr.State(value="")
                        # also expose sample_choice_s2 placeholders for single-tab (hidden)
                        sample_choice_s2_state = gr.State(value=None)
                        use_sample_s2_state = gr.State(value=False)
                        auto_fill_s2_state = gr.State(value=False)

                run_single_btn.click(
                    fn=infer_from_ui,
                    inputs=[
                        json_file,
                        manual_text,
                        prompt_wav,
                        sample_choice,
                        use_sample,
                        prompt_text,
                        speaker2_prompt_wav_state,  # speaker2 prompt wav (hidden state)
                        speaker2_prompt_text_state,  # speaker2 prompt text (hidden state)
                        dialogue_text_state,  # dialogue_text (hidden state)
                        use_dialect_prompt,
                        dialect_prompt_text,
                        model_choice,
                        model_path,
                        output_path,
                        llm_engine,
                        fp16_flow,
                        seed,
                    ],
                    outputs=[status, out_audio, model_status],
                )

            with gr.TabItem("Dual Speaker"):
                with gr.Row():
                    with gr.Column():
                        json_file_d = gr.File(label="Upload podcast JSON (optional) — follows example/podcast_script schema")
                        # For dual speaker workflows we prefer a dialogue textarea
                        dialogue_text = gr.Textbox(label="Dialogue (optional) — prefix lines with S1: or S2:", lines=8, placeholder="S1: Hello\nS2: Hi")
                        # allow two reference audios and texts
                        prompt_wav_d = gr.Audio(label="Reference audio for Speaker 1 (optional)", type="filepath")
                        prompt_text_d = gr.Textbox(label="Reference text for Speaker 1 (optional)")
                        speaker2_prompt_wav = gr.Audio(label="Reference audio for Speaker 2 (optional)", type="filepath")
                        speaker2_prompt_text = gr.Textbox(label="Reference text for Speaker 2 (optional)")
                        use_dialect_prompt_d = gr.Checkbox(label="Enable dialect prompt (use dialect-specific model)", value=False)
                        dialect_prompt_text_d = gr.Textbox(label="Dialect prompt (e.g. <|Henan|>我今天很高兴)", placeholder="<|Henan|>...", lines=2)

                    with gr.Column():
                        model_choice_d = gr.Dropdown(label="Select a local model (from pretrained_models)", choices=model_dirs, value=default_model)
                        model_path_d = gr.Textbox(label="Local model directory (override)", value=(os.path.join("pretrained_models", default_model) if default_model else "pretrained_models/SoulX-Podcast-1.7B-dialect"))
                        output_path_d = gr.Textbox(label="Output WAV file (will be overwritten)", value="outputs/output_dual.wav")
                        llm_engine_d = gr.Dropdown(label="LM engine (hf = HuggingFace, vllm = vLLM)", choices=llm_choices, value=llm_default)
                        fp16_flow_d = gr.Checkbox(label="Use FP16 for flow model (recommended on supported GPUs)", value=True)
                        seed_d = gr.Number(label="Random seed (int)", value=42, precision=0)
                        # per-speaker sample selectors and auto-fill
                        sample_choice_s1 = gr.Dropdown(label="Sample for Speaker 1 (repo examples)", choices=sample_files, value=(sample_files[0] if sample_files else None))
                        use_sample_s1 = gr.Checkbox(label="Use selected sample for Speaker 1 if no upload", value=True)
                        sample_choice_s2 = gr.Dropdown(label="Sample for Speaker 2 (repo examples)", choices=sample_files, value=(sample_files[1] if len(sample_files) > 1 else (sample_files[0] if sample_files else None)))
                        use_sample_s2 = gr.Checkbox(label="Use selected sample for Speaker 2 if no upload", value=False)
                        auto_fill_s2 = gr.Checkbox(label="Auto-fill Speaker 2 with Speaker 1 sample if Speaker 2 missing", value=True)
                        run_dual_btn = gr.Button("Run Dual-Speaker Inference")
                        status_d = gr.Textbox(label="Job status / messages", interactive=False)
                        out_audio_d = gr.Audio(label="Play or download generated audio", type="filepath")
                        model_status_d = gr.Textbox(label="Model status (loaded path)", interactive=False)

                        # Hidden/default state values for dual-tab invocation
                        manual_text_unused = gr.State(value="")
                        sample_choice_unused = gr.State(value=None)
                        use_sample_unused = gr.State(value=False)

                        # (Resolved sample path displays removed for both speakers)

                run_dual_btn.click(
                    fn=infer_from_ui,
                    inputs=[
                        json_file_d,
                        manual_text_unused,  # manual_text unused for dual
                        prompt_wav_d,
                        sample_choice_s1,
                        use_sample_s1,
                        prompt_text_d,
                        speaker2_prompt_wav,
                        speaker2_prompt_text,
                        dialogue_text,
                        use_dialect_prompt_d,
                        dialect_prompt_text_d,
                        model_choice_d,
                        model_path_d,
                        output_path_d,
                        llm_engine_d,
                        fp16_flow_d,
                        seed_d,
                        sample_choice_s2,
                        use_sample_s2,
                        auto_fill_s2,
                    ],
                    outputs=[status_d, out_audio_d, model_status_d],
                )

    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
