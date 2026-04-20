#!/usr/bin/env python3
"""Summarize JSONL traces emitted by serve_mosa_rag.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosa_rag.tracing import load_jsonl, summarize_traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Mosa RAG JSONL trace logs.")
    parser.add_argument("trace_path", type=Path, help="Path to a RAG_TRACE_PATH JSONL file")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.trace_path)
    summary = summarize_traces(rows)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Trace rows: {summary['count']}")
    print(f"Abstention rate: {summary['abstention_rate']:.2%}")
    print(f"Warn rate: {summary['warn_rate']:.2%}")
    print(f"Average latency: {summary['average_latency_ms']:.2f} ms")
    print(f"Answer modes: {summary['answer_modes']}")

    print("\nTop queries")
    for query, count in summary["top_queries"]:
        print(f"- {count}x {query}")

    print("\nTop result IDs")
    for result_id, count in summary["top_result_ids"]:
        print(f"- {count}x {result_id}")


if __name__ == "__main__":
    main()
