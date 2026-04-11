"""Helpers for retrieving records from normalized_mosa_rag.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from retrieve_pdf import Chunk


def load_rows(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        record = json.loads(raw)
        if not (record.get("retrieval_text") or "").strip():
            raise ValueError(f"line {line_number}: empty retrieval_text in {jsonl_path}")
        rows.append(record)

    if not rows:
        raise ValueError(f"no searchable records with non-empty retrieval_text in {jsonl_path}")
    return rows


def build_chunks(rows: list[dict], jsonl_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, record in enumerate(rows):
        chunks.append(
            Chunk(
                chunk_id=index,
                source_name=str(record.get("source_file", "")),
                source_path=str(jsonl_path.resolve()),
                page_number=int(record.get("source_page") or 0),
                text=(record.get("retrieval_text") or "").strip(),
            )
        )
    return chunks
