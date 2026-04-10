"""Build normalized_mosa_rag.jsonl from source PDFs in data/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosa_rag.extract_bar import extract_bar_records
from mosa_rag.extract_handbook import extract_handbook_records
from mosa_rag.extract_kitchen_batch import extract_kitchen_batch_records
from mosa_rag.extract_kitchen_open import extract_kitchen_open_records
from mosa_rag.extract_towels import extract_towel_records
from mosa_rag.pdf_text import read_pages
from mosa_rag.schema import Record

# Canonical source filenames in this repo (user docs may name the handbook differently).
DEFAULT_SOURCES: dict[str, str] = {
    "bar": "SOP - Bar (1).pdf",
    "kitchen_open": "SOP - Kitchen Open.pdf",
    "kitchen": "SOP - Kitchen.pdf",
    "handbook": "Mosa Employee Handbook.pdf",
    "towels": "Cleaning Towels Color Code.pdf",
}


def _assign_ids(records: list[Record]) -> None:
    used: set[str] = set()
    for i, r in enumerate(records, start=1):
        base = f"mosa_{i:05d}_{r.type}"
        rid = base
        n = 1
        while rid in used:
            n += 1
            rid = f"{base}_{n}"
        used.add(rid)
        r.id = rid


def build_records(data_dir: Path) -> list[Record]:
    records: list[Record] = []

    p_bar = data_dir / DEFAULT_SOURCES["bar"]
    records.extend(extract_bar_records(read_pages(p_bar), p_bar.name))

    p_ko = data_dir / DEFAULT_SOURCES["kitchen_open"]
    records.extend(extract_kitchen_open_records(read_pages(p_ko), p_ko.name))

    p_k = data_dir / DEFAULT_SOURCES["kitchen"]
    records.extend(extract_kitchen_batch_records(read_pages(p_k), p_k.name))

    p_hb = data_dir / DEFAULT_SOURCES["handbook"]
    records.extend(extract_handbook_records(read_pages(p_hb), p_hb.name))

    p_tw = data_dir / DEFAULT_SOURCES["towels"]
    records.extend(extract_towel_records(read_pages(p_tw), p_tw.name))

    _assign_ids(records)
    return records


def write_jsonl(records: list[Record], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in records]
    out_path.write_text("".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build normalized Mosa RAG JSONL corpus from PDFs.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing source PDFs (default: ./data)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("normalized_mosa_rag.jsonl"),
        help="Output JSONL path (default: ./normalized_mosa_rag.jsonl)",
    )
    args = ap.parse_args(argv)

    recs = build_records(args.data_dir)
    write_jsonl(recs, args.out)
    print(f"Wrote {len(recs)} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
