"""Local born-digital PDF fallback when GROBID is unavailable.

This extractor is intentionally conservative. GROBID remains the preferred backend
for academic structure; this path exists so the document workflow can be exercised
on machines such as Google Colab without running a separate GROBID service.
"""
from __future__ import annotations

import re
from pathlib import Path

from .cleaner import CleaningOptions, _remove_residual_citations
from .speech_normalizer import normalize_for_speech


_REFERENCE_HEADINGS = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited)\s*$", re.I
)
_FOOTNOTE_LINE = re.compile(r"^\s*\d{1,3}\s+\S+")
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.I)


def _clean_block(text: str, *, remove_inline_citations: bool) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = text.replace("\n", " ")
    if remove_inline_citations:
        text = _remove_residual_citations(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def extract_pdf_text_fallback(
    pdf_path: str | Path,
    *,
    options: CleaningOptions | None = None,
) -> str:
    """Extract reviewable speech text locally with PyMuPDF.

    Best for born-digital papers. It removes a References/Bibliography tail when a
    standalone heading is detected, repeated headers/footers, bare page numbers,
    conservative inline citations, and likely short numeric footnote lines.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "Local PDF fallback requires PyMuPDF. Install it with: pip install pymupdf"
        ) from exc

    options = options or CleaningOptions()
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    doc = fitz.open(path)
    pages: list[list[str]] = []
    for page in doc:
        blocks = []
        for block in page.get_text("blocks"):
            text = str(block[4]).strip()
            if text:
                blocks.append(text)
        pages.append(blocks)
    doc.close()

    # Repeated short blocks at page edges are usually running headers/footers.
    counts: dict[str, int] = {}
    for blocks in pages:
        for text in blocks[:2] + blocks[-2:]:
            key = re.sub(r"\s+", " ", text).strip()
            if len(key) <= 160:
                counts[key] = counts.get(key, 0) + 1
    repeated = {text for text, count in counts.items() if count >= 3}

    output: list[str] = []
    stop = False
    for blocks in pages:
        for raw in blocks:
            compact = re.sub(r"\s+", " ", raw).strip()
            if compact in repeated or _PAGE_NUMBER.fullmatch(compact):
                continue
            if _REFERENCE_HEADINGS.fullmatch(compact):
                stop = True
                break
            # Numeric footnotes are commonly small blocks beginning with a marker.
            # Keep longer blocks because numbered body lists are legitimate prose.
            if options.remove_notes and _FOOTNOTE_LINE.match(compact) and len(compact) < 240:
                continue
            cleaned = _clean_block(raw, remove_inline_citations=options.remove_inline_citations)
            if cleaned:
                output.append(normalize_for_speech(cleaned))
        if stop:
            break

    return "\n\n".join(output).strip()
