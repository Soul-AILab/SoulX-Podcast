import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

SENTENCE_ENDINGS = {".", "!", "?", "。", "！", "？"}
TRAILING_CLOSERS = set('"\'”’）)]}')
COMMON_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.",
    "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.", "no.", "st.",
}

# Silence inserted between consecutive sentence clips when they are joined into
# one track. Without it, back-to-back clips sound clipped.
DEFAULT_SEGMENT_GAP_MS = 150.0


@dataclass
class ScriptSentence:
    sentence_id: int
    speaker: str
    text: str
    chunk_id: int = -1
    estimate_seconds: float = 0.0


SPEAKER_PATTERN = re.compile(r"(\[S[1-4]\])")


def _is_likely_abbreviation(text: str, period_index: int) -> bool:
    if period_index <= 0:
        return False

    # First dot in dotted abbreviations (e.g., "p.m.", "U.S.")
    if period_index + 2 < len(text):
        if text[period_index + 1].isalpha() and text[period_index + 2] == ".":
            return True

    left = text[max(0, period_index - 16): period_index + 1]
    word_match = re.search(r"([A-Za-z]{1,12}\.)$", left)
    if word_match and word_match.group(1).lower() in COMMON_ABBREVIATIONS:
        return True

    acronym_match = re.search(r"(?:[A-Za-z]\.){2,}$", left)
    if acronym_match:
        return True

    if period_index > 0 and period_index + 1 < len(text):
        if text[period_index - 1].isdigit() and text[period_index + 1].isdigit():
            return True

    return False


def split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    start = 0
    idx = 0
    text_len = len(text)

    while idx < text_len:
        ch = text[idx]
        if ch in SENTENCE_ENDINGS:
            if ch == "." and _is_likely_abbreviation(text, idx):
                idx += 1
                continue

            end = idx + 1
            while end < text_len and text[end] in TRAILING_CLOSERS:
                end += 1

            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end
        idx += 1

    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def _parse_script_by_speaker(script_text: str, default_speaker: str = "S1") -> List[Tuple[str, str]]:
    script = script_text.strip()
    if not script:
        return []

    if not SPEAKER_PATTERN.search(script):
        return [(default_speaker, script)]

    turns: List[Tuple[str, str]] = []
    pieces = SPEAKER_PATTERN.split(script)
    current_speaker = None

    for piece in pieces:
        if not piece:
            continue
        if SPEAKER_PATTERN.fullmatch(piece):
            current_speaker = piece[1:-1]  # [S1] -> S1
            continue
        if current_speaker is None:
            current_speaker = default_speaker
        text = piece.strip()
        if text:
            turns.append((current_speaker, text))

    return turns


def estimate_sentence_duration_seconds(sentence: str) -> float:
    words = max(1, len(re.findall(r"\b\w+\b", sentence)))
    chars = max(1, len(sentence))
    punctuation_pauses = len(re.findall(r"[,;:，；：]", sentence)) * 0.12
    return max(words / 2.7, chars / 14.0) + punctuation_pauses


def split_script_into_sentence_units(
    script_text: str,
    max_chunk_seconds: float = 110.0,
    default_speaker: str = "S1",
) -> List[ScriptSentence]:
    if max_chunk_seconds <= 0:
        raise ValueError("max_chunk_seconds must be positive")

    turns = _parse_script_by_speaker(script_text, default_speaker=default_speaker)
    units: List[ScriptSentence] = []

    sentence_id = 0
    for speaker, turn_text in turns:
        for sentence in split_sentences(turn_text):
            estimate = estimate_sentence_duration_seconds(sentence)
            units.append(
                ScriptSentence(
                    sentence_id=sentence_id,
                    speaker=speaker,
                    text=sentence,
                    estimate_seconds=estimate,
                )
            )
            sentence_id += 1

    current_chunk = 0
    current_seconds = 0.0
    for unit in units:
        if unit.estimate_seconds > max_chunk_seconds:
            raise ValueError(
                f"Sentence {unit.sentence_id} is too long for max_chunk_seconds={max_chunk_seconds}. "
                "Please split this sentence manually before generation."
            )
        if current_seconds and current_seconds + unit.estimate_seconds > max_chunk_seconds:
            current_chunk += 1
            current_seconds = 0.0
        unit.chunk_id = current_chunk
        current_seconds += unit.estimate_seconds

    return units


def units_to_table(units: Sequence[ScriptSentence]) -> List[Dict[str, Any]]:
    return [
        {
            "sentence_id": unit.sentence_id,
            "chunk_id": unit.chunk_id,
            "speaker": unit.speaker,
            "text": unit.text,
            "estimate_seconds": round(unit.estimate_seconds, 2),
        }
        for unit in units
    ]


TABLE_HEADERS = ["sentence_id", "chunk_id", "speaker", "text", "estimate_seconds"]


def _normalize_header(value: Any) -> str:
    return str(value).strip().lower()


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if value.__class__.__name__ == "NAType":
        return True
    try:
        return bool(np.isnan(value))
    except Exception:
        pass
    try:
        unequal = value != value
    except Exception:
        return False
    return bool(unequal) if isinstance(unequal, (bool, np.bool_)) else False


def _is_blank_value(value: Any) -> bool:
    if _is_missing_value(value):
        return True
    return isinstance(value, str) and not value.strip()


def _row_from_values(values: Sequence[Any], headers: Sequence[Any] | None = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if headers:
        for i, header in enumerate(headers):
            key = _normalize_header(header)
            if key not in TABLE_HEADERS:
                continue
            if i >= len(values):
                continue
            value = values[i]
            if _is_missing_value(value):
                continue
            row[key] = value
        return row

    for i, key in enumerate(TABLE_HEADERS):
        if i >= len(values):
            continue
        value = values[i]
        if _is_missing_value(value):
            continue
        row[key] = value
    return row


def normalize_sentence_table_rows(table_value: Any) -> List[Dict[str, Any]]:
    if table_value is None:
        return []

    headers: Sequence[Any] | None = None
    rows_data: Any = None

    if hasattr(table_value, "headers") and hasattr(table_value, "data"):
        headers = getattr(table_value, "headers")
        rows_data = getattr(table_value, "data")
    elif isinstance(table_value, dict):
        if isinstance(table_value.get("data"), list):
            headers = table_value.get("headers")
            rows_data = table_value.get("data")
        elif table_value and all(isinstance(v, (list, tuple)) for v in table_value.values()):
            headers = list(table_value.keys())
            values = list(table_value.values())
            row_count = max(len(col) for col in values)
            rows_data = [
                [values[col_idx][row_idx] if row_idx < len(values[col_idx]) else None for col_idx in range(len(values))]
                for row_idx in range(row_count)
            ]
    elif hasattr(table_value, "to_dict"):
        try:
            records = table_value.to_dict("records")
            if isinstance(records, list):
                rows_data = records
        except TypeError:
            pass
        if rows_data is None:
            if hasattr(table_value, "columns") and hasattr(table_value, "values"):
                headers = list(getattr(table_value, "columns"))
                rows_data = getattr(table_value, "values")
            else:
                try:
                    dict_value = table_value.to_dict()
                except Exception:
                    dict_value = None
                if isinstance(dict_value, dict) and isinstance(dict_value.get("data"), list):
                    headers = dict_value.get("headers")
                    rows_data = dict_value.get("data")
    elif isinstance(table_value, list):
        rows_data = table_value

    if rows_data is None:
        return []

    if not isinstance(rows_data, list) and hasattr(rows_data, "tolist"):
        rows_data = rows_data.tolist()
    if isinstance(rows_data, tuple):
        rows_data = list(rows_data)

    rows: List[Dict[str, Any]] = []
    if isinstance(rows_data, list) and rows_data:
        first_row = rows_data[0]
        if isinstance(first_row, dict):
            for item in rows_data:
                row = {}
                for source_key, value in item.items():
                    key = _normalize_header(source_key)
                    if key not in TABLE_HEADERS:
                        continue
                    if _is_missing_value(value):
                        continue
                    row[key] = value
                rows.append(row)
        else:
            if headers is None and isinstance(first_row, (list, tuple)):
                first_headers = [_normalize_header(v) for v in first_row]
                if first_headers[: len(TABLE_HEADERS)] == TABLE_HEADERS:
                    headers = first_row
                    rows_data = rows_data[1:]
            for item in rows_data:
                if not isinstance(item, (list, tuple)):
                    continue
                rows.append(_row_from_values(item, headers=headers))

    while rows and all(_is_blank_value(rows[-1].get(key)) for key in TABLE_HEADERS):
        rows.pop()

    for i, row in enumerate(rows):
        text = row.get("text")
        has_other = any(not _is_blank_value(row.get(key)) for key in TABLE_HEADERS if key != "text")
        if _is_blank_value(text) and has_other:
            raise ValueError(
                f"Partially populated row {i}: missing text. "
                "Clear the row or provide sentence text."
            )

    return rows


def table_to_units(rows: Sequence[Dict[str, Any]]) -> List[ScriptSentence]:
    units: List[ScriptSentence] = []
    for i, row in enumerate(rows):
        speaker = str(row.get("speaker", "S1")).strip().upper()
        if not re.fullmatch(r"S[1-4]", speaker):
            raise ValueError(f"Invalid speaker at row {i}: {speaker}")

        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError(f"Empty sentence text at row {i}")

        sentence_id = int(row.get("sentence_id", i))
        chunk_id = int(row.get("chunk_id", -1))
        estimate = estimate_sentence_duration_seconds(text)
        units.append(
            ScriptSentence(
                sentence_id=sentence_id,
                chunk_id=chunk_id,
                speaker=speaker,
                text=text,
                estimate_seconds=estimate,
            )
        )

    units.sort(key=lambda item: item.sentence_id)
    return units


def group_sentence_ids_by_chunk(units: Sequence[ScriptSentence]) -> List[List[int]]:
    grouped: Dict[int, List[int]] = {}
    for unit in units:
        grouped.setdefault(unit.chunk_id, []).append(unit.sentence_id)
    return [grouped[key] for key in sorted(grouped.keys())]


def assemble_audio_and_manifest(
    segments: Sequence[Dict[str, Any]],
    sample_rate: int = 24000,
    gap_ms: float = DEFAULT_SEGMENT_GAP_MS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Join per-sentence clips into one track, with a short pause between them.

    ``gap_ms`` milliseconds of silence are inserted *between* consecutive clips
    -- never before the first or after the last -- so the joins do not sound
    clipped. The pause is part of the timeline, so ``start_seconds`` and
    ``end_seconds`` stay accurate for review and selective regeneration.
    """
    if gap_ms is None:
        gap_ms = 0.0
    gap_ms = float(gap_ms)
    if gap_ms < 0:
        raise ValueError("gap_ms must be >= 0")
    gap_samples = int(round(sample_rate * gap_ms / 1000.0))

    merged: List[np.ndarray] = []
    timeline: List[Dict[str, Any]] = []
    offset_samples = 0

    for index, segment in enumerate(
            sorted(segments, key=lambda item: item["sentence_id"])):
        audio = segment.get("audio")
        if audio is None:
            raise ValueError(f"Missing audio for sentence_id={segment['sentence_id']}")

        np_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration_samples = int(np_audio.shape[0])

        if index and gap_samples:
            merged.append(np.zeros((gap_samples,), dtype=np.float32))
            offset_samples += gap_samples

        timeline.append(
            {
                "sentence_id": int(segment["sentence_id"]),
                "chunk_id": int(segment.get("chunk_id", -1)),
                "speaker": segment["speaker"],
                "text": segment["text"],
                "start_sample": offset_samples,
                "end_sample": offset_samples + duration_samples,
                "start_seconds": round(offset_samples / sample_rate, 6),
                "end_seconds": round((offset_samples + duration_samples) / sample_rate, 6),
                "duration_seconds": round(duration_samples / sample_rate, 6),
            }
        )

        merged.append(np_audio)
        offset_samples += duration_samples

    final_audio = np.concatenate(merged) if merged else np.zeros((0,), dtype=np.float32)
    manifest = {
        "sample_rate": sample_rate,
        "total_duration_seconds": round(offset_samples / sample_rate, 6),
        "segments": timeline,
    }
    return final_audio, manifest


def replace_segment_audio(
    segments: Sequence[Dict[str, Any]],
    sentence_id: int,
    new_audio: np.ndarray,
) -> List[Dict[str, Any]]:
    updated = []
    replaced = False
    for segment in segments:
        cloned = dict(segment)
        if int(cloned["sentence_id"]) == int(sentence_id):
            cloned["audio"] = np.asarray(new_audio, dtype=np.float32).reshape(-1)
            replaced = True
        updated.append(cloned)

    if not replaced:
        raise ValueError(f"sentence_id={sentence_id} not found")
    return updated


def unit_to_dict(unit: ScriptSentence) -> Dict[str, Any]:
    return asdict(unit)
