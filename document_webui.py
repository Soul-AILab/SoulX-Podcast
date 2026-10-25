"""Web GUI for testing academic PDF -> speech-ready text.

This UI is intentionally independent of the TTS model so document ingestion can be
validated on Google Colab or a normal workstation without downloading/loading SoulX.
It is not Colab-specific; Gradio is simply launched with --share when remote access
is needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import gradio as gr

from soulxpodcast.document import CleaningOptions, extract_speech_text
from soulxpodcast.document.grobid_client import GrobidError
from soulxpodcast.document.pdf_fallback import extract_pdf_text_fallback


def _pdf_path(value) -> Path:
    if not value:
        raise ValueError("Please upload a PDF first.")
    path = Path(value)
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")
    return path


def extract_pdf(
    pdf_file,
    backend: str,
    grobid_url: str,
    include_title: bool,
    include_abstract: bool,
    include_headings: bool,
    remove_citations: bool,
):
    try:
        pdf_path = _pdf_path(pdf_file)
        options = CleaningOptions(
            include_title=include_title,
            include_abstract=include_abstract,
            include_headings=include_headings,
            remove_inline_citations=remove_citations,
        )

        selected = backend.lower()
        used_backend = selected
        note = ""

        if selected == "local (pymupdf)":
            text = extract_pdf_text_fallback(pdf_path, options=options)
            used_backend = "Local (PyMuPDF)"
        elif selected == "grobid":
            result = extract_speech_text(pdf_path, grobid_url=grobid_url, options=options)
            text = result.text
            used_backend = "GROBID"
        else:
            # Prefer GROBID for academic structure, but make remote testing zero-setup.
            try:
                result = extract_speech_text(pdf_path, grobid_url=grobid_url, options=options)
                text = result.text
                used_backend = "GROBID"
            except (GrobidError, OSError):
                text = extract_pdf_text_fallback(pdf_path, options=options)
                used_backend = "Local (PyMuPDF fallback)"
                note = " GROBID was unavailable, so the local fallback was used."

        if not text.strip():
            raise ValueError(
                "No readable text was extracted. The PDF may be scanned/image-only; OCR is not implemented yet."
            )

        output_dir = Path("outputs") / "document_text"
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = pdf_path.stem[:80]
        out = output_dir / f"{stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.speech.txt"
        out.write_text(text.rstrip() + "\n", encoding="utf-8")

        status = (
            f"Extraction complete using {used_backend}. "
            f"Produced {len(text):,} characters.{note} "
            "Review the text below before using it for speech generation."
        )
        return text, str(out), status
    except Exception as exc:
        return "", None, f"Extraction failed: {exc}"


def create_app():
    with gr.Blocks(title="SoulX-Podcast — Academic PDF to Text") as app:
        gr.Markdown(
            "## Academic PDF → Speech-ready Text\n"
            "Upload a born-digital academic PDF, remove non-spoken academic material, "
            "review/edit the result, and download a `.txt` file. TTS is intentionally not loaded here."
        )

        with gr.Row():
            pdf_input = gr.File(label="Academic PDF", file_types=[".pdf"], type="filepath")
            with gr.Column():
                backend = gr.Radio(
                    ["Auto", "Local (PyMuPDF)", "GROBID"],
                    value="Auto",
                    label="Extraction backend",
                    info="Auto prefers GROBID and falls back to local PyMuPDF.",
                )
                grobid_url = gr.Textbox(
                    value="http://localhost:8070",
                    label="GROBID URL",
                    info="Ignored by Local mode. You do not need GROBID for a smoke test.",
                )

        with gr.Accordion("Cleaning options", open=True):
            with gr.Row():
                include_title = gr.Checkbox(value=True, label="Keep title")
                include_abstract = gr.Checkbox(value=True, label="Keep abstract")
                include_headings = gr.Checkbox(value=True, label="Keep section headings")
                remove_citations = gr.Checkbox(value=True, label="Remove inline citations")

        extract_btn = gr.Button("Extract and clean text", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        # Keep component arguments conservative for compatibility across Gradio versions.
        text_output = gr.Textbox(
            label="Extracted text (editable review)",
            lines=28,
            interactive=True,
        )
        txt_file = gr.File(label="Download extracted .txt")

        extract_btn.click(
            fn=extract_pdf,
            inputs=[
                pdf_input,
                backend,
                grobid_url,
                include_title,
                include_abstract,
                include_headings,
                remove_citations,
            ],
            outputs=[text_output, txt_file, status],
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Academic PDF to text Gradio GUI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (useful for remote testing)")
    return parser.parse_args()


def main():
    args = parse_args()
    create_app().queue().launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
