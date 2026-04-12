#!/usr/bin/env python3
"""Retrieve context from normalized_mosa_rag.jsonl, then ask an LLM to answer from that context."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from mosa_rag.faiss_cache import default_cache_root, load_or_build_faiss_index
from mosa_rag.llm import DEFAULT_OLLAMA_MODEL, OLLAMA_LOCAL, OLLAMA_REMOTE, call_llm, resolve_ollama_generate_url
from mosa_rag.retrieve_jsonl import build_chunks, load_rows
from retrieve_pdf import MODEL_NAME, retrieve


def format_record(record: dict, score: float) -> str:
    lines = [
        f"id: {record.get('id', '')}",
        f"title: {record.get('title', '')}",
        f"type: {record.get('type', '')}",
        f"source: {record.get('source_file', '')} page {record.get('source_page', '')}",
        f"score: {score:.4f}",
    ]
    for field in ("retrieval_text", "rules", "steps", "ingredients", "storage_life", "threshold", "action"):
        value = record.get(field)
        if not value:
            continue
        if isinstance(value, list):
            joined = "; ".join(str(item) for item in value)
            lines.append(f"{field}: {joined}")
        else:
            lines.append(f"{field}: {value}")
    return "\n".join(lines)


def build_context(rows: list[dict], results: list[tuple[float, object]]) -> str:
    blocks: list[str] = []
    for rank, (score, chunk) in enumerate(results, start=1):
        record = rows[chunk.chunk_id]
        blocks.append(f"[record {rank}]\n{format_record(record, score)}")
    return "\n\n".join(blocks)


def build_user_input(query: str, context: str) -> str:
    return (
        "Using only the retrieved records below, write a clear internal staff answer in plain language.\n"
        "Tone: professional and direct—readable on a busy shift, but not chatty. Avoid rhetorical questions, "
        'filler ("so you want to", "here is the gist"), and generic workplace advice not found in the records.\n'
        "When you refer to a record, use the exact title from its `title:` line in the block below—not "
        "invented labels like 'record 1' or bracket numbers.\n"
        "If the records do not clearly answer the question, say that the retrieved records are insufficient. "
        "Do not invent steps, amounts, or policy; if a detail is missing from the records, say it is not in "
        "the retrieved records instead of guessing.\n"
        "End with a single line exactly in this form (use those same titles in Sources):\n"
        "Sources: <comma-separated record titles>\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved records:\n{context}"
    )


def _sync_llm_env_from_args(args: argparse.Namespace) -> None:
    os.environ["OLLAMA_PROVIDER"] = args.ollama_provider.strip()
    os.environ["OLLAMA_MODEL"] = (args.ollama_model or DEFAULT_OLLAMA_MODEL).strip()


def _validate_ollama_provider(name: str) -> None:
    if name not in (OLLAMA_LOCAL, OLLAMA_REMOTE):
        sys.exit(
            f"error: OLLAMA_PROVIDER / --ollama-provider must be {OLLAMA_LOCAL!r} or {OLLAMA_REMOTE!r}, got {name!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask an LLM to answer from retrieved Mosa JSONL records.")
    parser.add_argument("query")
    parser.add_argument("--jsonl", type=Path, default=Path("normalized_mosa_rag.jsonl"))
    parser.add_argument("--model-name", default=MODEL_NAME, help=f"Embedding model (default: {MODEL_NAME})")
    parser.add_argument("--top-k", type=int, default=5, help="How many retrieved records to send to the LLM")
    parser.add_argument("--raw-query", action="store_true", help="Omit BGE query instruction prefix")
    parser.add_argument(
        "--ollama-provider",
        default=os.getenv("OLLAMA_PROVIDER", ""),
        metavar="NAME",
        help=f"{OLLAMA_LOCAL} (http://localhost:11434/api/generate) or {OLLAMA_REMOTE} (OLLAMA_BASE_URL). "
        f"Default: OLLAMA_PROVIDER env.",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        help=f"Ollama model tag (default: {DEFAULT_OLLAMA_MODEL} or OLLAMA_MODEL)",
    )
    parser.add_argument("--show-context", action="store_true", help="Print retrieved records after the answer")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the API; print the prompt payload")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for FAISS cache (default: BGE_RAG_CACHE_DIR env, else .rag_index_cache next to --jsonl)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Always rebuild embeddings and FAISS index")
    return parser.parse_args()


def _resolve_cache_root(args: argparse.Namespace, jsonl: Path) -> Path | None:
    if args.no_cache:
        return None
    if args.cache_dir is not None:
        return args.cache_dir
    env = os.getenv("BGE_RAG_CACHE_DIR")
    if env:
        return Path(env)
    return default_cache_root(jsonl)


def main() -> None:
    args = parse_args()
    if not args.jsonl.is_file():
        sys.exit(f"error: corpus file not found: {args.jsonl}")

    try:
        rows = load_rows(args.jsonl)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    chunks = build_chunks(rows, args.jsonl)
    cache_root = _resolve_cache_root(args, args.jsonl)
    encoder, index, _ = load_or_build_faiss_index(
        chunks=chunks,
        model_name=args.model_name,
        source_jsonl=args.jsonl,
        cache_root=cache_root,
        no_cache=args.no_cache,
    )
    results = retrieve(
        encoder=encoder,
        index=index,
        chunks=chunks,
        query=args.query,
        top_k=min(args.top_k, len(chunks)),
        use_instruction=not args.raw_query,
    )
    context = build_context(rows, results)

    instructions = (
        "You are answering internal Mosa Tea staff questions. "
        "Only use the provided retrieved records—treat them as the only source of truth. "
        "If the records are insufficient or conflicting, say so plainly. "
        "Never present general knowledge or guesses as if they came from the records."
    )
    user_input = build_user_input(args.query, context)
    prompt = f"{instructions}\n\n{user_input}"

    _sync_llm_env_from_args(args)

    if args.dry_run:
        out: dict = {
            "ollama_provider": args.ollama_provider.strip() or None,
            "ollama_model": args.ollama_model,
            "prompt": prompt,
        }
        try:
            out["generate_url"] = resolve_ollama_generate_url()
        except RuntimeError as exc:
            out["generate_url_error"] = str(exc)
        print(json.dumps(out, indent=2))
        return

    if not args.ollama_provider.strip():
        sys.exit(
            f"error: set OLLAMA_PROVIDER to {OLLAMA_LOCAL} or {OLLAMA_REMOTE} "
            "(or pass --ollama-provider). Use --dry-run to inspect the prompt without calling Ollama."
        )

    _validate_ollama_provider(args.ollama_provider.strip())

    try:
        answer = call_llm(prompt)
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    if not answer:
        sys.exit("error: empty response from model")

    print("Answer")
    print("-" * 80)
    for i, block in enumerate(answer.strip().split("\n\n")):
        if i:
            print()
        print(textwrap.fill(block.strip(), width=100))

    if args.show_context:
        print("\nRetrieved Context")
        print("-" * 80)
        print(context)


if __name__ == "__main__":
    main()
