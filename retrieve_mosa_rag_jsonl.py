#!/usr/bin/env python3
"""Search normalized_mosa_rag.jsonl with BGE ONNX + FAISS (same as retrieve_pdf.py)."""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

from mosa_rag.faiss_cache import default_cache_root, load_or_build_faiss_index
from mosa_rag.retrieve_jsonl import build_chunks, load_rows
from retrieve_pdf import MODEL_NAME, Chunk, retrieve


def main() -> None:
    p = argparse.ArgumentParser(description="Search normalized_mosa_rag.jsonl with BGE + FAISS.")
    p.add_argument("query")
    p.add_argument("--jsonl", type=Path, default=Path("normalized_mosa_rag.jsonl"))
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--raw-query", action="store_true", help="Omit BGE query instruction prefix")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for FAISS cache (default: BGE_RAG_CACHE_DIR env, else .rag_index_cache next to --jsonl)",
    )
    p.add_argument("--no-cache", action="store_true", help="Always rebuild embeddings and FAISS index")
    a = p.parse_args()
    if not a.jsonl.is_file():
        sys.exit(f"error: corpus file not found: {a.jsonl}")

    try:
        rows = load_rows(a.jsonl)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    chunks: list[Chunk] = build_chunks(rows, a.jsonl)

    if a.no_cache:
        cache_root = None
    elif a.cache_dir is not None:
        cache_root = a.cache_dir
    elif os.getenv("BGE_RAG_CACHE_DIR"):
        cache_root = Path(os.environ["BGE_RAG_CACHE_DIR"])
    else:
        cache_root = default_cache_root(a.jsonl)

    print(f"Indexed {len(chunks)} records from {a.jsonl} | model {a.model_name}")
    enc, index, _ = load_or_build_faiss_index(
        chunks=chunks,
        model_name=a.model_name,
        source_jsonl=a.jsonl,
        cache_root=cache_root,
        no_cache=a.no_cache,
    )
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
