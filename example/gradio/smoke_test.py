"""Lightweight smoke test for the Gradio UI input handling.

This script performs a non-heavy check by creating a silent fallback reference audio
and validating that the example output path can be written. By default it DOES NOT
load or run the model. Use --run to perform an actual inference run (this may be
heavy and will load the selected model).

Usage:
    python example/gradio/smoke_test.py
    python example/gradio/smoke_test.py --run --model-path pretrained_models/SoulX-Podcast-1.7B

"""
import argparse
import os
import tempfile
import torch
import torchaudio
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def make_silent_wav(path: str, sr: int = 16000, duration_s: float = 0.5):
    samples = int(sr * duration_s)
    tensor = torch.zeros((1, samples), dtype=torch.float32)
    torchaudio.save(path, tensor, sr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true", help="If set, actually call the inference function (may be heavy)")
    p.add_argument("--model-path", type=str, default="", help="Model path to use when --run is set")
    p.add_argument("--output", type=str, default=os.path.join(OUT_DIR, "smoke_test_output.wav"))
    args = p.parse_args()

    print("Smoke test: creating a silent fallback reference audio and verifying output write")
    tmp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        make_silent_wav(tmp_ref.name)
        print(f"Created silent reference: {tmp_ref.name}")
    except Exception as e:
        print("Failed to create silent wav:", e)
        sys.exit(2)

    # Verify writing an output using the same format
    try:
        make_silent_wav(args.output)
        print(f"Wrote smoke output to: {args.output}")
    except Exception as e:
        print("Failed to write smoke output:", e)
        sys.exit(2)

    if args.run:
        # Attempt to run full inference via the gradio_app helper. This will load the model
        # and can be slow/heavy. We import lazily so the default smoke test remains fast.
        try:
            sys.path.insert(0, ROOT)
            import gradio_app as ga

            print("Running full inference (this may be slow). Model path:", args.model_path or "(default)")
            status, out, model_path = ga.infer_from_ui(
                None,  # json_file
                "[S1]Hello from smoke test",  # manual_text
                tmp_ref.name,  # prompt_wav_file
                None,  # sample_choice
                False,  # use_sample
                "",  # prompt_text
                None,  # speaker2 prompt wav
                "",  # speaker2 prompt text
                "",  # dialogue_text
                False,  # use_dialect_prompt
                "",  # dialect_prompt_text
                "",  # model_choice
                args.model_path or "",  # model_path
                args.output,
                "hf",
                True,
                42,
            )
            print("Inference result:", status, out, model_path)
        except Exception as e:
            print("Full inference failed:", e)
            sys.exit(3)

    print("Smoke test completed successfully")


if __name__ == "__main__":
    main()
