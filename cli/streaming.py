import os
import re
import json
import argparse
from copy import deepcopy

import soundfile as sf
import torch
import numpy as np
import sys
from pathlib import Path

# Ensure project root is on sys.path so local package imports work when
# running the script directly from the repository folder.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Default prompt sample (female) included in the repo
DEFAULT_PROMPT_REL = Path("example/audios/female_mandarin.wav")
DEFAULT_PROMPT_WAV = str((_REPO_ROOT / DEFAULT_PROMPT_REL))

from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.prompt_cache import PROMPT_FEATURE_CACHE
from soulxpodcast.utils.timing import TIMING_COLLECTOR, Timer
import tempfile
import atexit
import concurrent.futures
from datetime import datetime


def split_into_sentences_with_speaker(tagged_text: str):
    """Split a string like "[S1]Hello world. How are you?" into
    the list ["[S1]Hello world.", "[S1]How are you?"] keeping the speaker tag.

    Handles English punctuation and common Chinese punctuation.
    """
    pattern = r"^(\[S[1-9]\])(.+)$"
    m = re.match(pattern, tagged_text)
    if not m:
        # fallback: return as-is
        return [tagged_text]

    tag, text = m.group(1), m.group(2).strip()
    # Split on sentence enders, keep the delimiter
    # Handles: . ! ? and Chinese 。 ！ ？ and ellipses
    splitter = re.compile(r"(.+?)([\.\!\?…。！？]+)(\s*|$)")
    parts = []
    pos = 0
    for match in splitter.finditer(text):
        sent = (match.group(1) + match.group(2)).strip()
        if sent:
            parts.append(f"{tag}{sent}")
        pos = match.end()

    # leftover tail
    if pos < len(text):
        tail = text[pos:].strip()
        if tail:
            parts.append(f"{tag}{tail}")

    # If nothing matched, return original full
    if not parts:
        return [tag + text]

    return parts


def chunk_texts_at_sentence_level(text_list):
    """Given a list of tagged texts (each like '[S1]...'), return a flat
    list where each element is a single sentence with its speaker tag.
    """
    out = []
    for t in text_list:
        out.extend(split_into_sentences_with_speaker(t))
    return out


def play_audio_block(arr: np.ndarray, samplerate: int) -> bool:
    """Try to play back a numpy float array using available playback libraries.
    Returns True if played, False otherwise.
    """
    try:
        import sounddevice as sd
        sd.play(arr, samplerate)
        sd.wait()
        return True
    except Exception:
        try:
            import simpleaudio as sa
            pcm = (np.clip(arr, -1, 1) * 32767).astype(np.int16)
            sa.play_buffer(pcm.tobytes(), 1, 2, samplerate).wait_done()
            return True
        except Exception:
            print("[WARN] No audio playback libraries available (install sounddevice or simpleaudio)")
            return False


def run_streaming_inference(
    *,
    json_path: str | None,
    text: str | None,
    speaker: str,
    prompt_wav: str | None,
    prompt_text: str | None,
    model_path: str,
    output_path: str,
    llm_engine: str = "hf",
    fp16_flow: bool = False,
    seed: int = 1989,
    samplerate: int = 24000,
    reference_wav: str | None = None,
    play: bool = False,
    precache_prompts: bool = False,
    clear_cache: bool = False,
):
    # Build `inputs` either from json_path or from supplied text + prompt args
    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        inputs = podcast_format_parser(raw)
    else:
        # live text mode: construct flattened inputs dict used by downstream logic
        if text is None:
            raise ValueError("Either --json_path or --text must be provided")

        # If user provided plain lines, split them into turns by newline
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        # If no explicit speaker tag, prefix with supplied speaker
        formatted_texts = []
        for l in lines:
            if l.startswith("[S") and "]" in l:
                formatted_texts.append(l)
            else:
                formatted_texts.append(f"[{speaker}]" + l)

        inputs = {
            "key": "live",
            "prompt_text": [prompt_text or ""],
            "prompt_wav": [prompt_wav or ""],
            "text": formatted_texts,
            "spk": [0],
            "wav": output_path,
            "use_dialect_prompt": False,
            "dialect_prompt_text": [""],
        }

    # If a reference wav is provided, overwrite prompt_wav entries so
    # the model will use the provided reference sample(s).
    if reference_wav:
        # apply the same reference to all speakers if multiple exist
        inputs["prompt_wav"] = [reference_wav for _ in inputs.get("prompt_text", [])]
        print(f"[INFO] Using reference sample: {reference_wav} for all speakers")

    # If no prompt wav provided, prefer the repo female sample; otherwise create a
    # short silent WAV as a safe fallback.
    _temp_files = []
    if not any(inputs.get("prompt_wav", [])):
        if Path(DEFAULT_PROMPT_WAV).exists():
            inputs["prompt_wav"] = [DEFAULT_PROMPT_WAV]
            print(f"[INFO] No prompt wav supplied — using repo default sample: {DEFAULT_PROMPT_WAV}")
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            prompt_sr = 16000
            silence = np.zeros(int(prompt_sr * 0.4), dtype=np.float32)  # 400ms silence at 16k
            sf.write(tmp.name, silence, prompt_sr)
            inputs["prompt_wav"] = [tmp.name]
            _temp_files.append(tmp.name)
            print(f"[INFO] No prompt wav supplied — using generated silent prompt: {tmp.name}")

    # Sentence-level chunking of inputs['text']
    sentences = chunk_texts_at_sentence_level(inputs["text"])

    # Init model and dataset once
    def ui(msg: str):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[SYSTEM {ts}] {msg}")

    ui("Loading model (this may take a while)...")
    model, dataset = initiate_model(seed, model_path, llm_engine, fp16_flow)
    ui("Model loaded — ready for inference.")

    # record total run time when --timings is enabled
    if TIMING_COLLECTOR.enabled:
        total_timer = Timer('total_run')
        total_timer.__enter__()

    # Handle cache control
    if clear_cache:
        PROMPT_FEATURE_CACHE.clear()
        print("[cache] cleared prompt feature cache")

    if precache_prompts:
        prompts = [p for p in inputs.get("prompt_wav", []) if p]
        to_pre = [p for p in prompts if PROMPT_FEATURE_CACHE.get(p) is None]
        if to_pre:
            ui(f"Precomputing prompt features for {len(to_pre)} prompt(s)...")
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _precache_task(pth: str):
                try:
                    ds_item = {"key": "__precache__", "prompt_text": [""], "prompt_wav": [pth], "text": [""], "spk": [0], "wav": ""}
                    dataset.update_datasource([ds_item])
                    _ = dataset[0]
                    return (pth, True, None)
                except Exception as e:
                    return (pth, False, str(e))

            max_workers = min(4, len(to_pre))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_precache_task, p): p for p in to_pre}
                for fut in as_completed(futures):
                    pth, ok, err = fut.result()
                    if ok:
                        print(f"[cache] precomputed prompt features: {pth}")
                    else:
                        print(f"[cache] failed to precache {pth}: {err}")
            ui("Precompute complete.")

    # Prepare output directory if we will save to a file
    save_to_file = not play
    if save_to_file:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf_handle = sf.SoundFile(output_path, mode="w", samplerate=samplerate, channels=1, subtype="FLOAT")
        print(f"[INFO] Streaming inference will write to: {output_path}")
    else:
        sf_handle = None
        print(f"[INFO] Streaming inference will play back output live (no file save) if playback libs are available")

    # playback will use the module-level `play_audio_block(arr, samplerate)` helper

    def _compute_sentence(temp_inputs, sent):
        # Prepare data and run model to produce numpy arrays for generated wavs
        data = process_single_input(
            dataset,
            [sent],
            temp_inputs['prompt_wav'],
            temp_inputs['prompt_text'],
            temp_inputs['use_dialect_prompt'],
            temp_inputs.get('dialect_prompt_text', []),
        )
        # model inference timing
        with Timer('model_forward'):
            results = model.forward_longform(**data)
        out_arrs = []
        for wav in results.get("generated_wavs", []):
            arr = wav.cpu().squeeze(0).numpy()
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            out_arrs.append(arr)
        return out_arrs

    # Use a single-worker thread to precompute the next sentence while current plays
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {}
        # seed compute for the first sentence
        if len(sentences) > 0:
            futures[0] = executor.submit(_compute_sentence, deepcopy(inputs), sentences[0])

        for idx in range(len(sentences)):
            sent = sentences[idx]
            print(f"[INFO] Synthesizing sentence {idx+1}/{len(sentences)}: {sent}")

            # ensure current result is ready
            if idx in futures:
                out_arrs = futures[idx].result()
            else:
                # compute synchronously if not prefetched
                out_arrs = _compute_sentence(deepcopy(inputs), sent)

            # kick off next compute (prefetch)
            nxt = idx + 1
            if nxt < len(sentences) and nxt not in futures:
                futures[nxt] = executor.submit(_compute_sentence, deepcopy(inputs), sentences[nxt])

            # playback/write the results for current sentence
            for arr in out_arrs:
                if play:
                    played = play_audio_block(arr, samplerate)
                    if not played and sf_handle is None:
                        print("[WARN] Playback failed; consider installing sounddevice/simpleaudio or allow saving to file")
                if sf_handle is not None:
                    sf_handle.write(arr)
                    sf_handle.flush()

    if sf_handle is not None:
        sf_handle.close()
    # cleanup any generated temp files
    for f in _temp_files:
        try:
            os.remove(f)
        except Exception:
            pass

    # finish total run timer if enabled
    if TIMING_COLLECTOR.enabled:
        try:
            total_timer.__exit__(None, None, None)
        except Exception:
            pass

    print(f"[INFO] Streaming synthesis completed{'' if not save_to_file else ' — output at: ' + output_path}")


def synthesize_sentences(model, dataset, inputs, sentences, sf_handle, play, samplerate, precache_prompts: bool = False, clear_cache: bool = False):
    """Synthesize a list of sentences (each with speaker tag) using provided model/dataset.
    Writes to sf_handle if provided and plays back if play=True.
    """
    def _compute_sentence_local(temp_inputs, sent):
        try:
            data = process_single_input(
                dataset,
                [sent],
                temp_inputs['prompt_wav'],
                temp_inputs['prompt_text'],
                temp_inputs['use_dialect_prompt'],
                temp_inputs.get('dialect_prompt_text', []),
            )
        except Exception as e:
            print(f"[ERROR] Failed to process input for sentence '{sent}': {e}")
            return []
        with Timer('model_forward'):
            results = model.forward_longform(**data)
        out = []
        for wav in results.get("generated_wavs", []):
            arr = wav.cpu().squeeze(0).numpy()
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            out.append(arr)
        return out

    # Optionally handle cache control for REPL path
    if clear_cache:
        PROMPT_FEATURE_CACHE.clear()
        print("[cache] cleared prompt feature cache")

    if precache_prompts:
        prompts = [p for p in inputs.get("prompt_wav", []) if p]
        to_pre = [p for p in prompts if PROMPT_FEATURE_CACHE.get(p) is None]
        if to_pre:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _precache_task(pth: str):
                try:
                    ds_item = {"key": "__precache__", "prompt_text": [""], "prompt_wav": [pth], "text": [""], "spk": [0], "wav": ""}
                    dataset.update_datasource([ds_item])
                    _ = dataset[0]
                    return (pth, True, None)
                except Exception as e:
                    return (pth, False, str(e))

            max_workers = min(4, len(to_pre))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_precache_task, p): p for p in to_pre}
                for fut in as_completed(futures):
                    pth, ok, err = fut.result()
                    if ok:
                        print(f"[cache] precomputed prompt features: {pth}")
                    else:
                        print(f"[cache] failed to precache {pth}: {err}")

    # Precompute next sentence while current plays to reduce gaps
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {}
        if len(sentences) > 0:
            futures[0] = executor.submit(_compute_sentence_local, deepcopy(inputs), sentences[0])

        for idx in range(len(sentences)):
            sent = sentences[idx]
            print(f"[INFO] Synthesizing sentence {idx+1}/{len(sentences)}: {sent}")

            # ensure we have outputs for current
            if idx in futures:
                out_arrs = futures[idx].result()
            else:
                out_arrs = _compute_sentence_local(deepcopy(inputs), sent)

            # prefetch next
            nxt = idx + 1
            if nxt < len(sentences) and nxt not in futures:
                futures[nxt] = executor.submit(_compute_sentence_local, deepcopy(inputs), sentences[nxt])

            for arr in out_arrs:
                if play:
                    played = play_audio_block(arr, samplerate)
                    if not played and sf_handle is None:
                        print("[WARN] Playback failed; consider installing sounddevice/simpleaudio or allow saving to file")
                if sf_handle is not None:
                    sf_handle.write(arr)
                    sf_handle.flush()


def interactive_repl(model_path: str, output_path: str, llm_engine: str, fp16_flow: bool, seed: int, prompt_wav: str | None, prompt_text: str | None, reference_wav: str | None, play: bool, samplerate: int, precache_prompts: bool = False, clear_cache: bool = False):
    """Start an interactive REPL that accepts input without blocking model work.

    This function launches an input reader thread that pushes user lines into a
    queue and a worker thread that consumes queued lines and synthesizes them.
    Enter an empty line to quit the session.
    """
    import threading
    import queue

    def ui(msg: str):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[SYSTEM {ts}] {msg}")

    ui("Loading model (this may take a while)...")
    model, dataset = initiate_model(seed, model_path, llm_engine, fp16_flow)
    ui("Model loaded — ready for interactive REPL.")

    base_inputs = {
        "key": "live",
        "prompt_text": [prompt_text or ""],
        "prompt_wav": [prompt_wav or ""],
        "text": [],
        "spk": [0],
        "wav": output_path,
        "use_dialect_prompt": False,
        "dialect_prompt_text": [""],
    }

    # Apply reference sample if present
    if reference_wav:
        base_inputs["prompt_wav"] = [reference_wav]

    # If no prompt wav provided, prefer the repo female sample; otherwise create a
    # short silent WAV as a safe fallback.
    _temp_files = []
    if not any(base_inputs.get("prompt_wav", [])):
        if Path(DEFAULT_PROMPT_WAV).exists():
            base_inputs["prompt_wav"] = [DEFAULT_PROMPT_WAV]
            print(f"[INFO] No prompt wav supplied — using repo default sample: {DEFAULT_PROMPT_WAV}")
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            prompt_sr = 16000
            silence = np.zeros(int(prompt_sr * 0.4), dtype=np.float32)  # 400ms silence at 16k
            sf.write(tmp.name, silence, prompt_sr)
            base_inputs["prompt_wav"] = [tmp.name]
            _temp_files.append(tmp.name)
            print(f"[INFO] No prompt wav supplied — using generated silent prompt: {tmp.name}")

    save_to_file = not play
    sf_handle = None
    if save_to_file:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf_handle = sf.SoundFile(output_path, mode="w", samplerate=samplerate, channels=1, subtype="FLOAT")

    line_queue: "queue.Queue[str|None]" = queue.Queue()

    def input_reader():
        """Read user lines and push into the queue. Empty line => stop signal."""
        try:
            while True:
                try:
                    line = input(">>> ")
                except EOFError:
                    line = ""
                if line is None:
                    break
                line = line.strip()
                # empty line signals quit
                if line == "":
                    line_queue.put(None)
                    break
                line_queue.put(line)
                # echo queue confirmation so user knows their input was captured
                try:
                    qsize = line_queue.qsize()
                    print(f"[queued] {line} (queue={qsize})")
                except Exception:
                    print(f"[queued] {line}")
        finally:
            return

    stop_event = threading.Event()

    # play_queue will receive tuples (arr, samplerate) to playback/write
    play_queue: "queue.Queue[tuple[float, int] | None]" = queue.Queue()

    def compute_worker():
        """Consume queued lines, run model compute, and push audio blocks to play_queue."""
        try:
            while not stop_event.is_set():
                item = line_queue.get()
                if item is None:
                    # propagate termination to playback and exit
                    play_queue.put(None)
                    break

                inputs = dict(base_inputs)
                if item.startswith("[S") and "]" in item:
                    inputs["text"] = [item]
                else:
                    inputs["text"] = [f"[S1]{item}"]

                sentences = chunk_texts_at_sentence_level(inputs["text"])

                # For each sentence, compute audio frames and push them to play queue
                for sent in sentences:
                    try:
                        data = process_single_input(
                            dataset,
                            [sent],
                            inputs['prompt_wav'],
                            inputs['prompt_text'],
                            inputs['use_dialect_prompt'],
                            inputs.get('dialect_prompt_text', []),
                        )
                    except Exception as e:
                        print(f"[ERROR] Failed to prepare input for '{sent}': {e}")
                        continue

                    # run model (this is where GPU is used)
                    try:
                        # model inference timing
                        with Timer('model_forward'):
                            results = model.forward_longform(**data)
                    except Exception as e:
                        print(f"[ERROR] Model generation failed for '{sent}': {e}")
                        continue

                    for wav in results.get("generated_wavs", []):
                        arr = wav.cpu().squeeze(0).numpy()
                        if arr.ndim > 1:
                            arr = arr.reshape(-1)
                        # push to play queue for playback/writing
                        play_queue.put((arr, samplerate))

                line_queue.task_done()
        finally:
            stop_event.set()

    def playback_worker():
        """Consume generated audio blocks and play/write them sequentially."""
        try:
            while True:
                item = play_queue.get()
                if item is None:
                    break
                arr, sr = item
                if play:
                    played = play_audio_block(arr, sr)
                    if not played and sf_handle is None:
                        print("[WARN] Playback failed; consider installing sounddevice/simpleaudio or allow saving to file")
                if sf_handle is not None:
                    sf_handle.write(arr)
                    sf_handle.flush()
                play_queue.task_done()
        finally:
            return

    print("[INFO] Enter text lines to synthesize. Empty line = quit.")
    reader_t = threading.Thread(target=input_reader, daemon=True)
    compute_t = threading.Thread(target=compute_worker, daemon=True)
    playback_t = threading.Thread(target=playback_worker, daemon=True)
    reader_t.start()
    compute_t.start()
    playback_t.start()

    ui("REPL ready — you may type prompts. Inputs are queued and will synthesize in order.")

    # Wait for compute to finish, then for playback to consume remaining items
    try:
        compute_t.join()
        # wait until playback thread sees the termination sentinel and finishes
        playback_t.join()
    except KeyboardInterrupt:
        stop_event.set()
        try:
            line_queue.put_nowait(None)
        except Exception:
            pass

    # cleanup
    if sf_handle is not None:
        sf_handle.close()
    for f in _temp_files:
        try:
            os.remove(f)
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input modes: either --json_path OR --text (live)
    parser.add_argument("--json_path", help="Path to the input JSON file (optional)")
    parser.add_argument("--text", help="Live input text (use newlines for multiple turns)")
    parser.add_argument("--speaker", default="S1", help="Default speaker tag to use for live text (e.g. S1)")
    parser.add_argument("--prompt_wav", default="", help="Path to prompt wav for live text (optional)")
    parser.add_argument("--prompt_text", default="", help="Prompt text for live text (optional)")
    parser.add_argument("--reference_wav", default=None, help="Optional reference sample to apply to all speakers")

    parser.add_argument("--model_path", required=True, help="Path to the model directory")
    parser.add_argument("--output_path", default="outputs/stream_out.wav", help="Path to the output audio file")
    parser.add_argument("--llm_engine", default="hf", choices=["hf", "vllm"], help="Inference engine to use")
    parser.add_argument("--fp16_flow", action="store_true", help="Enable FP16 flow")
    parser.add_argument("--play", action="store_true", help="Play back generated audio live (no file save by default)")
    parser.add_argument("--repl", action="store_true", help="Start an interactive REPL waiting for text input")
    parser.add_argument("--samplerate", type=int, default=24000, help="Output samplerate (Hz)")
    parser.add_argument("--seed", type=int, default=1988, help="Random seed")
    parser.add_argument("--precache_prompts", action="store_true", help="Precompute features for prompt WAVs at startup")
    parser.add_argument("--clear_cache", action="store_true", help="Clear prompt feature cache before running")
    parser.add_argument("--timings", action="store_true", help="Enable per-stage timing logs and report summary")
    args = parser.parse_args()

    # Enable timing collector if requested
    if args.timings:
        TIMING_COLLECTOR.reset()
        TIMING_COLLECTOR.enabled = True

    # If user asked for REPL explicitly, or provided neither json nor text, start interactive REPL
    if args.repl or (not args.json_path and not args.text):
        interactive_repl(
            model_path=args.model_path,
            output_path=args.output_path,
            llm_engine=args.llm_engine,
            fp16_flow=args.fp16_flow,
            seed=args.seed,
            prompt_wav=args.prompt_wav or None,
            prompt_text=args.prompt_text or None,
            reference_wav=args.reference_wav,
            play=args.play,
            samplerate=args.samplerate,
            precache_prompts=args.precache_prompts,
            clear_cache=args.clear_cache,
        )
    else:
        run_streaming_inference(
            json_path=args.json_path,
            text=args.text,
            speaker=args.speaker,
            prompt_wav=args.prompt_wav,
            prompt_text=args.prompt_text,
            model_path=args.model_path,
            output_path=args.output_path,
            llm_engine=args.llm_engine,
            fp16_flow=args.fp16_flow,
            seed=args.seed,
            samplerate=args.samplerate,
            reference_wav=args.reference_wav,
            play=args.play,
            precache_prompts=args.precache_prompts,
            clear_cache=args.clear_cache,
        )

    # If timings were enabled, print the collected report
    if args.timings:
        print('\n')
        TIMING_COLLECTOR.report()
