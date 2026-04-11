#!/usr/bin/env python3
"""Search normalized_mosa_rag.jsonl with BGE ONNX + FAISS (same as retrieve_pdf.py)."""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from retrieve_pdf import MODEL_NAME, Chunk, build_faiss_index, retrieve


def main() -> None:
    p = argparse.ArgumentParser(description="Search normalized_mosa_rag.jsonl with BGE + FAISS.")
    p.add_argument("query")
    p.add_argument("--jsonl", type=Path, default=Path("normalized_mosa_rag.jsonl"))
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--raw-query", action="store_true", help="Omit BGE query instruction prefix")
    a = p.parse_args()
    if not a.jsonl.is_file():
        sys.exit(f"error: corpus file not found: {a.jsonl}")

    rows: list[dict] = []
    for n, line in enumerate(a.jsonl.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"warning: line {n}: invalid JSON ({e})", file=sys.stderr)
            continue
        if not (rec.get("retrieval_text") or "").strip():
            print(f"warning: line {n}: empty retrieval_text, skipped", file=sys.stderr)
            continue
        rows.append(rec)

    if not rows:
        sys.exit("error: no searchable records with non-empty retrieval_text")

    chunks: list[Chunk] = []
    for i, rec in enumerate(rows):
        t = (rec.get("retrieval_text") or "").strip()
        chunks.append(
            Chunk(i, str(rec.get("source_file", "")), str(a.jsonl.resolve()), int(rec.get("source_page") or 0), t)
        )

    print(f"Indexed {len(chunks)} records from {a.jsonl} | model {a.model_name}")
    enc, index, _ = build_faiss_index(chunks=chunks, model_name=a.model_name)
    for rank, (score, ch) in enumerate(
        retrieve(enc, index, chunks, a.query, min(a.top_k, len(chunks)), not a.raw_query), 1
    ):
        r = rows[ch.chunk_id]
        print("=" * 96, f"\nrank={rank} score={score:.4f}", sep="")
        print(f"id={r.get('id','')} type={r.get('type','')} title={r.get('title','')}")
        print(f"source_file={r.get('source_file','')} source_page={r.get('source_page','')}")
        print("-" * 96)
        print(textwrap.fill(str(r.get("retrieval_text", "")), width=96), "\n", sep="")


if __name__ == "__main__":
    main()
