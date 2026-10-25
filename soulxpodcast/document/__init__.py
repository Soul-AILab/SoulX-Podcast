"""Academic-document ingestion utilities for speech generation."""

from .cleaner import CleaningOptions, clean_tei_for_speech
from .grobid_client import GrobidClient, GrobidError
from .pipeline import SpeechDocument, extract_speech_text
from .speech_normalizer import normalize_for_speech
from .tei_parser import ParsedDocument, parse_tei

__all__ = [
    "CleaningOptions",
    "GrobidClient",
    "GrobidError",
    "ParsedDocument",
    "SpeechDocument",
    "clean_tei_for_speech",
    "extract_speech_text",
    "normalize_for_speech",
    "parse_tei",
]
