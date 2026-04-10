"""Shared record shape and allowed record types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ALLOWED_TYPES: frozenset[str] = frozenset(
    {
        "drink_recipe",
        "batch_prep_recipe",
        "prep_threshold_rule",
        "opening_checklist_item",
        "closing_checklist_item",
        "inventory_minimum",
        "pos_procedure",
        "policy_rule",
        "cleaning_rule",
    }
)


@dataclass
class Record:
    id: str
    type: str
    title: str
    entity_name: str = ""
    doc_type: str = ""
    role_scope: list[str] = field(default_factory=list)
    shift_scope: list[str] = field(default_factory=list)
    day_scope: list[str] = field(default_factory=list)
    time_scope: list[str] = field(default_factory=list)
    ingredients: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    storage_life: str = ""
    threshold: str = ""
    action: str = ""
    tags: list[str] = field(default_factory=list)
    source_file: str = ""
    source_page: int = 0
    retrieval_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        out: dict[str, Any] = {}
        for k, v in d.items():
            if v == "" or v == []:
                continue
            out[k] = v
        for req in ("id", "type", "title", "retrieval_text"):
            out[req] = d[req]
        return out
