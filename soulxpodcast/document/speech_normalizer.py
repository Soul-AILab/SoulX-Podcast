from __future__ import annotations

import re


_ABBREVIATIONS = {
    r"\bFig\.\s*": "Figure ",
    r"\bFigs\.\s*": "Figures ",
    r"\bEq\.\s*": "Equation ",
    r"\bEqs\.\s*": "Equations ",
}


def normalize_for_speech(text: str) -> str:
    """Conservative normalization before TTS.

    This intentionally avoids rewriting equations or scientific notation until
    their pronunciation policy is explicitly defined and tested.
    """
    for pattern, replacement in _ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\bet\s+al\.\b", "and colleagues", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
