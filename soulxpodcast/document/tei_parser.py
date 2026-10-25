from __future__ import annotations

from dataclasses import dataclass, field
import re
import xml.etree.ElementTree as ET


@dataclass
class Section:
    heading: str = ""
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    title: str = ""
    abstract: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _children(element: ET.Element, name: str):
    return [child for child in element if _local(child.tag) == name]


def _first_descendant(root: ET.Element, name: str) -> ET.Element | None:
    for element in root.iter():
        if _local(element.tag) == name:
            return element
    return None


def parse_tei(tei_xml: str) -> ParsedDocument:
    """Parse the useful reading structure from GROBID TEI XML.

    Bibliography lives outside TEI ``body`` and is intentionally ignored.
    """
    root = ET.fromstring(tei_xml)
    document = ParsedDocument()

    title_stmt = _first_descendant(root, "titleStmt")
    if title_stmt is not None:
        title = next((e for e in title_stmt.iter() if _local(e.tag) == "title"), None)
        document.title = _text(title)

    profile = _first_descendant(root, "profileDesc")
    if profile is not None:
        abstract = next((e for e in profile.iter() if _local(e.tag) == "abstract"), None)
        if abstract is not None:
            document.abstract = [_text(p) for p in abstract.iter() if _local(p.tag) == "p" and _text(p)]

    body = _first_descendant(root, "body")
    if body is None:
        return document

    direct_divs = _children(body, "div")
    if not direct_divs:
        paragraphs = [_text(p) for p in body.iter() if _local(p.tag) == "p" and _text(p)]
        if paragraphs:
            document.sections.append(Section(paragraphs=paragraphs))
        return document

    for div in direct_divs:
        heading = ""
        head = next((e for e in div if _local(e.tag) == "head"), None)
        if head is not None:
            heading = _text(head)
        paragraphs = [_text(p) for p in div.iter() if _local(p.tag) == "p" and _text(p)]
        if heading or paragraphs:
            document.sections.append(Section(heading=heading, paragraphs=paragraphs))

    return document
