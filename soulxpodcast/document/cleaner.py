from __future__ import annotations

from dataclasses import dataclass
import re
import xml.etree.ElementTree as ET

from .tei_parser import ParsedDocument, Section, parse_tei


@dataclass(frozen=True)
class CleaningOptions:
    include_title: bool = True
    include_abstract: bool = True
    include_headings: bool = True
    remove_figures: bool = True
    remove_tables: bool = True
    remove_notes: bool = True
    remove_inline_citations: bool = True


_DROP_TAGS = {"figure", "table", "note"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _remove_element_keep_tail(parent: ET.Element, child: ET.Element) -> None:
    tail = child.tail or ""
    index = list(parent).index(child)
    if index > 0:
        previous = list(parent)[index - 1]
        previous.tail = (previous.tail or "") + tail
    else:
        parent.text = (parent.text or "") + tail
    parent.remove(child)


def _strip_unspoken_xml(root: ET.Element, options: CleaningOptions) -> None:
    for parent in list(root.iter()):
        for child in list(parent):
            tag = _local(child.tag)
            should_drop = (
                (tag == "figure" and options.remove_figures)
                or (tag == "table" and options.remove_tables)
                or (tag == "note" and options.remove_notes)
            )
            if tag == "ref" and options.remove_inline_citations:
                ref_type = child.attrib.get("type", "")
                should_drop = ref_type in {"bibr", "footnote"}
            if should_drop:
                _remove_element_keep_tail(parent, child)
            else:
                _strip_unspoken_xml(child, options)


def _remove_residual_citations(text: str) -> str:
    # Numeric styles: [1], [1, 3, 7], [4-8]. Avoid broad bracket removal.
    text = re.sub(r"\s*\[(?:\d+[a-z]?\s*(?:[-–,;]\s*\d+[a-z]?\s*)*)\]", "", text, flags=re.I)
    # Parenthetical author-year citations. Conservative: requires a four-digit year.
    text = re.sub(
        r"\s*\((?:[A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][A-Za-z'’-]+)?\s*,?\s*)?(?:19|20)\d{2}[a-z]?(?:\s*;[^()]*?(?:19|20)\d{2}[a-z]?)*\)",
        "",
        text,
    )
    # Narrative citation: Smith et al. (2021) -> Smith et al.
    text = re.sub(r"\s*\((?:19|20)\d{2}[a-z]?\)", "", text)
    return text


def _clean_text(text: str, remove_inline_citations: bool) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)  # common PDF line-break hyphenation
    if remove_inline_citations:
        text = _remove_residual_citations(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def clean_tei_for_speech(tei_xml: str, options: CleaningOptions | None = None) -> ParsedDocument:
    options = options or CleaningOptions()
    root = ET.fromstring(tei_xml)
    _strip_unspoken_xml(root, options)
    document = parse_tei(ET.tostring(root, encoding="unicode"))

    if not options.include_title:
        document.title = ""
    if not options.include_abstract:
        document.abstract = []

    document.title = _clean_text(document.title, options.remove_inline_citations)
    document.abstract = [_clean_text(p, options.remove_inline_citations) for p in document.abstract]
    cleaned_sections: list[Section] = []
    for section in document.sections:
        heading = _clean_text(section.heading, options.remove_inline_citations) if options.include_headings else ""
        paragraphs = [_clean_text(p, options.remove_inline_citations) for p in section.paragraphs]
        paragraphs = [p for p in paragraphs if p]
        if heading or paragraphs:
            cleaned_sections.append(Section(heading=heading, paragraphs=paragraphs))
    document.sections = cleaned_sections
    return document
