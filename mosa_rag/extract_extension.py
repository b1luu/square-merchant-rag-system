"""Extension.pdf -> supplemental records for policy notes, procedures, and recipes."""

from __future__ import annotations

import re

from mosa_rag.normalize import collapse_whitespace, normalize_for_parse
from mosa_rag.pdf_text import PageText
from mosa_rag.schema import Record

SECTION_HEADING_RE = r"(?:\[(?:policy|procedure|recipe)\]\s*)?[A-Z][A-Za-z0-9/&()' -]{0,120}:?"
SECTION_RE = re.compile(
    rf"(?P<title>{SECTION_HEADING_RE})\s*●\s*(?P<body>.*?)(?=(?:{SECTION_HEADING_RE}\s*●)|\Z)"
)
TYPE_HINT_RE = re.compile(r"^\[(?P<hint>policy|procedure|recipe)\]\s*", re.IGNORECASE)

DRINK_HINTS = (
    "shaker",
    "tea",
    "creamer",
    "ice",
    "serve",
    "strain",
    "scoop",
    "ml",
    "recipe",
)
PROCEDURE_HINTS = (
    "square",
    "pos",
    "clock in",
    "clock out",
    "schedule",
    "availability",
    "request time off",
    "passcode",
)
TAG_KEYWORDS = {
    "paid sick leave": "paid_sick_leave",
    "sick": "sick",
    "leave": "leave",
    "illness": "illness",
    "tgy": "tgy",
    "hun kue": "hun_kue",
    "square": "square",
    "pos": "pos",
}


def _clean_title(title: str) -> str:
    return collapse_whitespace(title).strip(" :-")


def _split_bullets(body: str) -> list[str]:
    return [piece for piece in (collapse_whitespace(part) for part in body.split("●")) if piece]


def _fallback_title(text: str) -> str:
    prefix, sep, _ = text.partition(":")
    if sep and len(prefix.split()) <= 8:
        return _clean_title(prefix)
    words = text.split()
    return _clean_title(" ".join(words[:6])) or "Extension note"


def _iter_sections(text: str) -> list[tuple[str, list[str]]]:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        stripped = collapse_whitespace(text)
        return [(_fallback_title(stripped), [stripped])] if stripped else []

    sections: list[tuple[str, list[str]]] = []
    for match in matches:
        title = _clean_title(match.group("title"))
        bullets = _split_bullets(match.group("body"))
        if title and bullets:
            sections.append((title, bullets))
    return sections


def _classify_section(title: str, bullets: list[str]) -> tuple[str, str]:
    hint_match = TYPE_HINT_RE.match(title)
    if hint_match:
        hint = hint_match.group("hint").lower()
        cleaned_title = _clean_title(TYPE_HINT_RE.sub("", title))
        return (
            {
                "policy": "policy_rule",
                "procedure": "pos_procedure",
                "recipe": "drink_recipe",
            }[hint],
            cleaned_title,
        )

    text = f"{title} {' '.join(bullets)}".lower()
    if any(token in text for token in DRINK_HINTS):
        return "drink_recipe", title
    if any(token in text for token in PROCEDURE_HINTS):
        return "pos_procedure", title
    return "policy_rule", title


def _build_tags(title: str, bullets: list[str]) -> list[str]:
    tags = ["extension"]
    haystack = f"{title} {' '.join(bullets)}".lower()
    for needle, tag in TAG_KEYWORDS.items():
        if needle in haystack and tag not in tags:
            tags.append(tag)
    return tags


def _build_retrieval_text(rtype: str, title: str, bullets: list[str]) -> str:
    joined = " ".join(bullets)
    if rtype == "drink_recipe":
        steps = " ".join(f"Step {i}: {step}" for i, step in enumerate(bullets, start=1))
        return f"{title}. Type: drink_recipe. Extension recipe note. {steps}"
    if rtype == "pos_procedure":
        steps = " ".join(f"Step {i}: {step}" for i, step in enumerate(bullets, start=1))
        return f"{title}. Type: pos_procedure. Extension procedure note. {steps}"
    return f"{title}. Type: policy_rule. Extension policy note. {joined}"


def extract_extension_records(pages: list[PageText], source_file: str) -> list[Record]:
    records: list[Record] = []

    for page in pages:
        page_text = normalize_for_parse(page.text).replace("•", "●")
        for raw_title, bullets in _iter_sections(page_text):
            rtype, title = _classify_section(raw_title, bullets)
            records.append(
                Record(
                    id="",
                    type=rtype,
                    title=title,
                    entity_name=title,
                    doc_type="extension",
                    role_scope=["all_staff"] if rtype != "drink_recipe" else ["bar", "kitchen"],
                    rules=bullets if rtype == "policy_rule" else [],
                    steps=bullets if rtype != "policy_rule" else [],
                    tags=_build_tags(title, bullets),
                    source_file=source_file,
                    source_page=page.page_number,
                    retrieval_text=_build_retrieval_text(rtype, title, bullets),
                )
            )

    return records
