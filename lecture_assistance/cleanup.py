from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List


_whitespace_re = re.compile(r"\s+")
_repeat_words_re = re.compile(r"\b(\w+)(?:\s+\1\b)+", flags=re.IGNORECASE)
_punct_space_re = re.compile(r"\s+([?.!,;:])")
_non_alnum_keep_space_re = re.compile(r"[^a-zA-Z0-9\s?.!,;:'/-]+")


def normalize_for_compare(text: str) -> str:
    """Normalization used for duplicate detection."""
    text = text.lower().strip()
    text = _non_alnum_keep_space_re.sub(" ", text)
    text = _whitespace_re.sub(" ", text)
    return text


def clean_ocr_line(text: str) -> str:
    """Clean one OCR line/fragment into a readable sentence fragment."""
    if not text:
        return ""

    text = text.replace("\x0c", " ")
    text = text.replace("\u200b", " ")
    text = text.strip()

    # Remove obvious OCR noise and compact whitespace.
    text = _whitespace_re.sub(" ", text)

    # Remove duplicated adjacent words created by OCR noise.
    text = _repeat_words_re.sub(r"\1", text)

    # Keep common punctuation but strip the junk around the edges.
    text = text.strip(" \t-_|•·~`")

    # Fix spacing before punctuation.
    text = _punct_space_re.sub(r"\1", text)

    # Drop very noisy fragments.
    if len(text) < 2:
        return ""

    return text


def clean_ocr_lines(lines: Iterable[str]) -> List[str]:
    """Clean a list of OCR outputs and remove repeated entries inside the batch."""
    cleaned: List[str] = []
    seen = set()

    for line in lines:
        item = clean_ocr_line(line)
        if not item:
            continue

        key = normalize_for_compare(item)
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(item)

    return cleaned


def similarity(a: str, b: str) -> float:
    """String similarity for duplicate suppression."""
    return SequenceMatcher(None, normalize_for_compare(a), normalize_for_compare(b)).ratio()


def is_duplicate(candidate: str, recent_items: Iterable[str], threshold: float = 0.88) -> bool:
    """Return True if the candidate is too similar to any recently accepted item."""
    candidate_norm = normalize_for_compare(candidate)
    if not candidate_norm:
        return True

    for item in recent_items:
        if not item:
            continue
        if similarity(candidate, item) >= threshold:
            return True

    return False


def join_display_lines(lines: Iterable[str]) -> str:
    """Join unique lines into a clean display string."""
    joined = "\n".join([line.strip() for line in lines if line and line.strip()])
    return joined.strip()
