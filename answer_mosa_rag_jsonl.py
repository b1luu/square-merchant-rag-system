#!/usr/bin/env python3
"""Retrieve context from normalized_mosa_rag.jsonl, then ask an LLM to answer from that context."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from mosa_rag.llm import DEFAULT_OLLAMA_MODEL, OLLAMA_LOCAL, OLLAMA_REMOTE, call_llm, resolve_ollama_generate_url
from mosa_rag.runtime import ResidentRetriever
from retrieve_pdf import MODEL_NAME


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

def main() -> None:
    args = parse_args()
    if not args.jsonl.is_file():
        sys.exit(f"error: corpus file not found: {args.jsonl}")

    try:
        retriever = ResidentRetriever(
            jsonl=args.jsonl,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
        )
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    prompt, context, _ = retriever.build_prompt_bundle(
        args.query,
        top_k=args.top_k,
        raw_query=args.raw_query,
    )

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
