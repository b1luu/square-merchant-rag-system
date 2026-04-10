#!/usr/bin/env python3
"""
Semantic search over normalized_mosa_rag.jsonl using the same BGE ONNX + FAISS stack as retrieve_pdf.py.

Example:
  python retrieve_mosa_rag_jsonl.py "How do I make boba in the rice cooker?"
  python retrieve_mosa_rag_jsonl.py "Square POS passcode" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from retrieve_pdf import MODEL_NAME, Chunk, build_faiss_index, retrieve


def load_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def records_to_chunks(records: list[dict], jsonl_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for rec in records:
        text = (rec.get("retrieval_text") or "").strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=len(chunks),
                source_name=str(rec.get("source_file", "")),
                source_path=str(jsonl_path),
                page_number=int(rec.get("source_page") or 0),
                text=text,
            )
        )
    if not chunks:
        raise ValueError("No non-empty retrieval_text entries in JSONL.")
    return chunks


def attach_meta(chunks: list[Chunk], records: list[dict]) -> dict[int, dict]:
    """Map chunk_id -> full record for display (ids may skip if retrieval_text empty)."""
    meta: dict[int, dict] = {}
    chunk_i = 0
    for rec in records:
        if not (rec.get("retrieval_text") or "").strip():
            continue
        meta[chunk_i] = rec
        chunk_i += 1
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the normalized Mosa RAG JSONL corpus with BGE + FAISS."
    )
    parser.add_argument(
        "query",
        help="Natural-language question or keywords",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("normalized_mosa_rag.jsonl"),
        help="Path to normalized_mosa_rag.jsonl (default: ./normalized_mosa_rag.jsonl)",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=f"BGE model id (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of records to return (default: 5)",
    )
    parser.add_argument(
        "--raw-query",
        action="store_true",
        help="Do not prepend the BGE retrieval instruction to the query",
    )
    args = parser.parse_args()

    if not args.jsonl.is_file():
        raise SystemExit(
            f"JSONL not found: {args.jsonl}\n"
            "Build it first: python -m mosa_rag.build_corpus --data-dir data --out normalized_mosa_rag.jsonl"
        )

    records = load_records(args.jsonl)
    chunks = records_to_chunks(records, args.jsonl)
    meta = attach_meta(chunks, records)

    print(f"Loaded {len(chunks)} records from {args.jsonl}")
    print(f"Embedding with {args.model_name} (ONNX) …")

    encoder, index, _ = build_faiss_index(chunks=chunks, model_name=args.model_name)
    results = retrieve(
        encoder=encoder,
        index=index,
        chunks=chunks,
        query=args.query,
        top_k=min(args.top_k, len(chunks)),
        use_instruction=not args.raw_query,
    )

    print(f"\nQuery: {args.query}\nTop {len(results)} matches:\n")

    for rank, (score, chunk) in enumerate(results, start=1):
        rec = meta.get(chunk.chunk_id, {})
        rid = rec.get("id", "?")
        rtype = rec.get("type", "?")
        title = rec.get("title", "?")
        print("=" * 100)
        print(
            f"Rank {rank} | score={score:.4f} | id={rid} | type={rtype}\n"
            f"Title: {title}\n"
            f"Source: {chunk.source_name} | page={chunk.page_number}"
        )
        print("-" * 100)
        print(textwrap.fill(chunk.text, width=100))
        print()


if __name__ == "__main__":
    main()
