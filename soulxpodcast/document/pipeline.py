from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cleaner import CleaningOptions, clean_tei_for_speech
from .grobid_client import GrobidClient
from .speech_normalizer import normalize_for_speech
from .tei_parser import ParsedDocument


@dataclass
class SpeechDocument:
    text: str
    document: ParsedDocument
    tei_xml: str


def _render_document(document: ParsedDocument) -> str:
    blocks: list[str] = []
    if document.title:
        blocks.append(document.title)
    if document.abstract:
        blocks.append("Abstract")
        blocks.extend(document.abstract)
    for section in document.sections:
        if section.heading:
            blocks.append(section.heading)
        blocks.extend(section.paragraphs)
    return "\n\n".join(block for block in blocks if block.strip())


def extract_speech_text(
    pdf_path: str | Path,
    *,
    grobid_url: str = "http://localhost:8070",
    options: CleaningOptions | None = None,
    client: GrobidClient | None = None,
) -> SpeechDocument:
    """Convert a born-digital academic PDF into reviewable TTS text."""
    client = client or GrobidClient(grobid_url)
    tei_xml = client.process_fulltext(pdf_path)
    document = clean_tei_for_speech(tei_xml, options=options)
    text = _render_document(document)
    # Normalize per paragraph so the preview remains editable and readable.
    text = "\n\n".join(normalize_for_speech(block) for block in text.split("\n\n") if block.strip())
    return SpeechDocument(text=text, document=document, tei_xml=tei_xml)
