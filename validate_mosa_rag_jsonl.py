#!/usr/bin/env python3
"""Validate normalized_mosa_rag.jsonl structure and basic integrity checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mosa_rag.schema import ALLOWED_TYPES


def validate(path: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    ids: set[str] = set()
    n_records = 0
    if not path.exists():
        return [f"File not found: {path}"], 0

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        n_records += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {lineno}: invalid JSON ({e})")
            continue

        for key in ("id", "type", "title", "retrieval_text"):
            if key not in obj or not isinstance(obj[key], str) or not obj[key].strip():
                errors.append(f"Line {lineno}: missing/empty string field {key!r}")

        rid = obj.get("id")
        if isinstance(rid, str) and rid:
            if rid in ids:
                errors.append(f"Line {lineno}: duplicate id {rid!r}")
            ids.add(rid)

        rtype = obj.get("type")
        if isinstance(rtype, str) and rtype and rtype not in ALLOWED_TYPES:
            errors.append(f"Line {lineno}: unknown type {rtype!r} (not in allowed set)")

    return errors, n_records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate Mosa RAG JSONL corpus.")
    ap.add_argument("jsonl", type=Path, nargs="?", default=Path("normalized_mosa_rag.jsonl"))
    args = ap.parse_args(argv)

    errs, n_records = validate(args.jsonl)
    if errs:
        for e in errs:
            print(e, file=sys.stderr)
        print(f"Validation FAILED ({len(errs)} issue(s)).", file=sys.stderr)
        return 1
    print(f"OK: {args.jsonl} passed validation ({n_records} records).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
