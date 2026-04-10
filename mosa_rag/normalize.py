"""Normalize PDF-extracted text for downstream parsing."""

from __future__ import annotations

import re


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_pdf_line_noise(text: str) -> str:
    """
    Fix common pypdf artifacts: double spaces, stray newlines mid-sentence.
    Keeps paragraph breaks as newlines when there are blank-line gaps in source.
    """
    # Unify line endings
    t = text.replace("\r", "\n")
    # Replace single newlines that are clearly intra-paragraph wraps
    t = re.sub(r"(?<=[a-zA-Z0-9%,])\n(?=[a-z])", " ", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def normalize_for_parse(text: str) -> str:
    """Single-line style normalization for regex parsers."""
    return collapse_whitespace(normalize_pdf_line_noise(text))
