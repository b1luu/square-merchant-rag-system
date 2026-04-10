"""Cleaning towel color code → cleaning_rule records."""

from __future__ import annotations

from mosa_rag.normalize import normalize_for_parse
from mosa_rag.pdf_text import PageText, join_pages
from mosa_rag.schema import Record


_TOWEL_BLOCKS = [
    (
        "Yellow (cleanest towel)",
        "yellow",
        [
            "Use the yellow towel to wipe the rim of drink cups and any spilled drink on the cup before serving.",
            "Yellow is only for ready-to-serve drinks (the cleanest towel).",
        ],
    ),
    (
        "Green (food-contact surfaces)",
        "green",
        [
            "Use the green towel to clean food-contact surfaces such as prep station counters and boba station counters.",
            "Green is for areas that directly handle food or drink ingredients (second cleanest).",
        ],
    ),
    (
        "Blue (general cleaning)",
        "blue",
        [
            "Use the blue towel for general surfaces such as marble countertops, kitchen sinks, work tables, and customer tables and chairs.",
            "Do not use blue towels on food-contact areas.",
        ],
    ),
    (
        "Red (restroom / storage — dirtiest)",
        "red",
        [
            "Use the red towel for restroom cleaning or storage shelves.",
            "Red towels are strictly not allowed in food or drink prep/serving areas.",
        ],
    ),
]


def extract_towel_records(pages: list[PageText], source_file: str) -> list[Record]:
    """Derive towel rules from PDF text; uses canonical wording (not raw PDF lines)."""
    full = normalize_for_parse(join_pages(pages))
    if "Cleaning Towel" not in full and "color" not in full.lower():
        raise ValueError("Unexpected towel PDF content")

    records: list[Record] = []
    for title, color, sentences in _TOWEL_BLOCKS:
        rt = (
            f"{title}. Type: cleaning_rule. Applies to bar and kitchen staff during cleaning and sanitizing tasks. "
            + " ".join(sentences)
        )
        records.append(
            Record(
                id="",  # filled by builder
                type="cleaning_rule",
                title=title,
                entity_name=f"{color} cleaning towel",
                doc_type="cleaning_guide",
                role_scope=["bar", "kitchen", "all_staff"],
                shift_scope=["opening", "mid", "closing"],
                rules=sentences,
                tags=["towel", "color_code", "sanitation", color],
                source_file=source_file,
                source_page=1,
                retrieval_text=rt,
            )
        )
    return records
