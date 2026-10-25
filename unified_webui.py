"""Unified SoulX-Podcast web GUI.

This is the canonical user-facing GUI for the PDF-to-audio workflow.  It combines
academic PDF ingestion and the existing segmented SoulX generation interface in one
Gradio application.  The application is not Colab-specific; ``--share`` is simply
useful when the process is running on a remote machine such as Colab.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

import colab_webui
import document_webui


MODEL_DIRNAME = "SoulX-Podcast-1.7B"
REQUIRED_MODEL_FILES = ("soulxpodcast_config.json", "flow.pt", "hift.pt")


def _model_dir_candidates(explicit):
    here = Path(__file__).resolve().parent
    if explicit:
        yield Path(explicit).expanduser()
    yield here / "pretrained_models" / MODEL_DIRNAME
    for sibling in sorted(here.parent.glob(
            f"SoulX-Podcast*/pretrained_models/{MODEL_DIRNAME}")):
        yield sibling


def _resolve_model_path(explicit=None):
    tried = []
    for candidate in _model_dir_candidates(explicit):
        candidate = candidate.resolve()
        tried.append(str(candidate))
        if not candidate.is_dir():
            continue
        missing = [name for name in REQUIRED_MODEL_FILES if not (candidate / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Model directory {candidate} is incomplete (missing {', '.join(missing)})."
            )
        return str(candidate)
    raise FileNotFoundError(
        "Could not find the SoulX-Podcast model directory. Looked in:\n  "
        + "\n  ".join(tried)
        + "\nDownload it with:\n"
        f"  hf download Soul-AILab/{MODEL_DIRNAME} --local-dir pretrained_models/{MODEL_DIRNAME}"
    )


def create_app(model_path: str, llm_engine: str = "hf", fp16_flow: bool = False, seed: int = 1988):
    """Build one Gradio app containing document preparation and speech generation."""
    # Build the existing apps once.  TTS model initialization still happens only
    # here at application startup; the PDF tab itself does not invoke the model.
    document_app = document_webui.create_app()
    tts_app = colab_webui.create_app(model_path, llm_engine, fp16_flow, seed)

    with gr.Blocks(title="SoulX Academic Paper Reader") as app:
        gr.Markdown(
            "# SoulX Academic Paper Reader\n"
            "Prepare an academic PDF as editable speech text, then use the same text "
            "with SoulX voice-cloned speech generation."
        )
        with gr.Tabs():
            with gr.TabItem("1. PDF → Text"):
                gr.Markdown(
                    "Extract and clean the paper first. Review the result and, if desired, "
                    "download the `.txt` file before moving to speech generation."
                )
                document_app.render()
            with gr.TabItem("2. Text → Audio"):
                gr.Markdown(
                    "Paste the reviewed text into **Initial Script**, provide the reference "
                    "voice and transcript, then split and generate audio."
                )
                tts_app.render()
    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SoulX academic PDF-to-audio web GUI")
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--llm_engine", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--fp16_flow", action="store_true")
    parser.add_argument("--seed", type=int, default=1988)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link for remote access")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = _resolve_model_path(args.model_path)
    app = create_app(model_path, args.llm_engine, args.fp16_flow, args.seed)
    app.queue().launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
