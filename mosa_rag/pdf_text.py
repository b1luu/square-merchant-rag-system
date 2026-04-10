"""Load per-page text from PDFs using pypdf."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based
    text: str


def read_pages(pdf_path: str | Path) -> list[PageText]:
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        pages.append(PageText(page_number=i, text=raw))
    return pages


def join_pages(pages: list[PageText], separator: str = "\n") -> str:
    return separator.join(p.text for p in pages)
