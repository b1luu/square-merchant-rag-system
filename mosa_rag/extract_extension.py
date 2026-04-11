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
    "clock in",
    "clock out",
    "schedule",
    "availability",
    "request time off",
    "passcode",
)
TAG_KEYWORDS = {
    "wifi": "wifi",
    "paid sick leave": "paid_sick_leave",
    "sick-hour accrual": "sick_hour_accrual",
    "sick": "sick",
    "leave": "leave",
    "paid family leave": "paid_family_leave",
    "bereavement": "bereavement",
    "health insurance": "health_insurance",
    "part-time": "part_time",
    "meal policy": "meal_policy",
    "refund": "refund",
    "override code": "override_code",
    "illness": "illness",
    "tgy": "tgy",
    "hun kue": "hun_kue",
    "square": "square",
}
KNOWN_SECTION_TITLES = (
    "What to do if you are sick?",
    "Sick-hour accrual",
    "TGY Special",
    "Wifi password:",
    "Paid Family Leave",
    "Bereavement leave:",
    "Health Insurance Benefits:",
    "Employee Meal Policy",
    "Square Refunds After Payment",
    "Override Code:",
)
TITLE_TYPE_OVERRIDES = {
    "What to do if you are sick?": "policy_rule",
    "Sick-hour accrual": "policy_rule",
    "TGY Special": "drink_recipe",
    "Wifi password": "policy_rule",
    "Paid Family Leave": "policy_rule",
    "Bereavement leave": "policy_rule",
    "Health Insurance Benefits": "policy_rule",
    "Employee Meal Policy": "policy_rule",
    "Square Refunds After Payment": "pos_procedure",
    "Override Code": "pos_procedure",
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


def _iter_known_sections(text: str) -> list[tuple[str, list[str]]]:
    lower_text = text.lower()
    matches: list[tuple[int, int, str]] = []
    for title in KNOWN_SECTION_TITLES:
        start = lower_text.find(title.lower())
        if start != -1:
            matches.append((start, start + len(title), title))

    if not matches:
        return []

    matches.sort(key=lambda item: item[0])
    sections: list[tuple[str, list[str]]] = []
    for idx, (_, end, raw_title) in enumerate(matches):
        next_start = matches[idx + 1][0] if idx + 1 < len(matches) else len(text)
        body = collapse_whitespace(text[end:next_start])
        if not body:
            continue
        bullets = _split_bullets(body) if "●" in body else [body]
        sections.append((_clean_title(raw_title), bullets))
    return sections


def _matches_hint(text: str, hint: str) -> bool:
    if re.fullmatch(r"[a-z0-9_-]+", hint):
        return re.search(rf"\b{re.escape(hint)}\b", text) is not None
    return hint in text


def _classify_section(title: str, bullets: list[str]) -> tuple[str, str]:
    if title in TITLE_TYPE_OVERRIDES:
        return TITLE_TYPE_OVERRIDES[title], title

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
    if any(_matches_hint(text, token) for token in DRINK_HINTS):
        return "drink_recipe", title
    if any(_matches_hint(text, token) for token in PROCEDURE_HINTS):
        return "pos_procedure", title
    return "policy_rule", title


def _build_tags(title: str, bullets: list[str]) -> list[str]:
    tags = ["extension"]
    haystack = f"{title} {' '.join(bullets)}".lower()
    for needle, tag in TAG_KEYWORDS.items():
        if _matches_hint(haystack, needle) and tag not in tags:
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
        sections = _iter_known_sections(page_text) or _iter_sections(page_text)
        for raw_title, bullets in sections:
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
