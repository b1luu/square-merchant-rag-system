#!/usr/bin/env python3
"""Evaluate retrieval quality against curated JSONL query sets for normalized_mosa_rag.jsonl."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from retrieve_pdf import MODEL_NAME, Chunk, build_faiss_index, retrieve

DEFAULT_JSONL = Path("normalized_mosa_rag.jsonl")
DEFAULT_CASE_FILES = (
    Path("eval_sets/mosa_rag_smoke.jsonl"),
    Path("eval_sets/mosa_rag_paraphrase.jsonl"),
)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    category: str
    expected_ids: tuple[str, ...]
    expected_titles: tuple[str, ...]
    notes: str = ""


def load_corpus(jsonl_path: Path) -> tuple[list[dict], list[Chunk]]:
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {jsonl_path}")

    rows: list[dict] = []
    chunks: list[Chunk] = []
    for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        record = json.loads(raw)
        retrieval_text = (record.get("retrieval_text") or "").strip()
        if not retrieval_text:
            raise ValueError(f"Line {line_number} has empty retrieval_text: {jsonl_path}")
        rows.append(record)
        chunks.append(
            Chunk(
                chunk_id=len(chunks),
                source_name=str(record.get("source_file", "")),
                source_path=str(jsonl_path.resolve()),
                page_number=int(record.get("source_page") or 0),
                text=retrieval_text,
            )
        )

    if not rows:
        raise ValueError(f"No searchable rows found in {jsonl_path}")
    return rows, chunks


def load_cases(case_path: Path) -> list[EvalCase]:
    if not case_path.is_file():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    cases: list[EvalCase] = []
    for line_number, line in enumerate(case_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        record = json.loads(raw)
        case_id = str(record.get("id") or f"{case_path.stem}:{line_number}")
        query = str(record.get("query") or "").strip()
        category = str(record.get("category") or "uncategorized").strip()
        expected_ids_raw = record.get("expected_ids") or []
        if isinstance(expected_ids_raw, str):
            expected_ids = (expected_ids_raw,)
        else:
            expected_ids = tuple(str(item) for item in expected_ids_raw)
        expected_titles_raw = record.get("expected_titles") or []
        if isinstance(expected_titles_raw, str):
            expected_titles = (expected_titles_raw,)
        else:
            expected_titles = tuple(str(item) for item in expected_titles_raw)
        if not query:
            raise ValueError(f"{case_path}:{line_number} is missing query")
        cases.append(
            EvalCase(
                case_id=case_id,
                query=query,
                category=category,
                expected_ids=expected_ids,
                expected_titles=expected_titles,
                notes=str(record.get("notes") or "").strip(),
            )
        )

    if not cases:
        raise ValueError(f"No cases found in {case_path}")
    return cases


def summarize_result_rows(rows: list[dict], results: list[tuple[float, Chunk]], limit: int = 3) -> list[str]:
    summary: list[str] = []
    for rank, (score, chunk) in enumerate(results[:limit], start=1):
        record = rows[chunk.chunk_id]
        summary.append(f"{rank}. {record['id']} | score={score:.4f} | {record['title']}")
    return summary


def evaluate_case_file(
    *,
    case_path: Path,
    rows: list[dict],
    chunks: list[Chunk],
    encoder: object,
    index: object,
    top_k: int,
    use_instruction: bool,
) -> None:
    cases = load_cases(case_path)
    print(f"\n=== {case_path} ===")
    print(f"Cases loaded: {len(cases)}")

    scored_cases: list[tuple[EvalCase, int | None, list[tuple[float, Chunk]]]] = []
    probe_cases: list[tuple[EvalCase, list[tuple[float, Chunk]]]] = []

    for case in cases:
        results = retrieve(
            encoder=encoder,
            index=index,
            chunks=chunks,
            query=case.query,
            top_k=min(top_k, len(chunks)),
            use_instruction=use_instruction,
        )
        if case.expected_ids:
            match_rank = None
            for rank, (_, chunk) in enumerate(results, start=1):
                record = rows[chunk.chunk_id]
                if record["id"] in case.expected_ids or record["title"] in case.expected_titles:
                    match_rank = rank
                    break
            scored_cases.append((case, match_rank, results))
        elif case.expected_titles:
            match_rank = None
            for rank, (_, chunk) in enumerate(results, start=1):
                if rows[chunk.chunk_id]["title"] in case.expected_titles:
                    match_rank = rank
                    break
            scored_cases.append((case, match_rank, results))
        else:
            probe_cases.append((case, results))

    if scored_cases:
        hit_at_1 = sum(1 for _, rank, _ in scored_cases if rank == 1)
        hit_at_k = sum(1 for _, rank, _ in scored_cases if rank is not None)
        mrr = sum(0.0 if rank is None else 1.0 / rank for _, rank, _ in scored_cases) / len(scored_cases)

        print(f"Scored cases: {len(scored_cases)}")
        print(f"Hit@1: {hit_at_1}/{len(scored_cases)}")
        print(f"Hit@{top_k}: {hit_at_k}/{len(scored_cases)}")
        print(f"MRR: {mrr:.3f}")

        category_rows: dict[str, list[int | None]] = defaultdict(list)
        for case, match_rank, _ in scored_cases:
            category_rows[case.category].append(match_rank)

        print("\nBy category")
        print("category cases hit@1 hit@k mrr")
        for category in sorted(category_rows):
            ranks = category_rows[category]
            category_hit_at_1 = sum(1 for rank in ranks if rank == 1)
            category_hit_at_k = sum(1 for rank in ranks if rank is not None)
            category_mrr = sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks)
            print(
                f"{category:<22} {len(ranks):>5} {category_hit_at_1:>5}/{len(ranks):<5} "
                f"{category_hit_at_k:>5}/{len(ranks):<5} {category_mrr:.3f}"
            )

        failures = [(case, rank, results) for case, rank, results in scored_cases if rank is None]
        soft_misses = [(case, rank, results) for case, rank, results in scored_cases if rank not in (None, 1)]
        if failures:
            print("\nMisses")
            for case, _, results in failures:
                print(f"- {case.case_id} [{case.category}] {case.query}")
                expected = ", ".join(case.expected_ids or case.expected_titles)
                print(f"  expected: {expected}")
                if case.notes:
                    print(f"  notes: {case.notes}")
                for row in summarize_result_rows(rows, results):
                    print(f"  got: {row}")
        else:
            print("\nMisses\n- none")

        if soft_misses:
            print("\nNot rank 1")
            for case, rank, results in soft_misses:
                print(f"- {case.case_id} [{case.category}] rank={rank} {case.query}")
                if case.notes:
                    print(f"  notes: {case.notes}")
                for row in summarize_result_rows(rows, results):
                    print(f"  got: {row}")
        else:
            print("\nNot rank 1\n- none")

    if probe_cases:
        print("\nManual probes")
        for case, results in probe_cases:
            print(f"- {case.case_id} [{case.category}] {case.query}")
            if case.notes:
                print(f"  notes: {case.notes}")
            for row in summarize_result_rows(rows, results):
                print(f"  top: {row}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate normalized_mosa_rag.jsonl against curated query sets.")
    parser.add_argument(
        "case_files",
        nargs="*",
        type=Path,
        default=list(DEFAULT_CASE_FILES),
        help="JSONL case files to run (defaults to smoke + paraphrase sets)",
    )
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL, help=f"Corpus path (default: {DEFAULT_JSONL})")
    parser.add_argument("--model-name", default=MODEL_NAME, help=f"Embedding model (default: {MODEL_NAME})")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k depth for success checks (default: 5)")
    parser.add_argument("--raw-query", action="store_true", help="Do not prepend the BGE query instruction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, chunks = load_corpus(args.jsonl)
    encoder, index, _ = build_faiss_index(chunks=chunks, model_name=args.model_name)
    print(f"Corpus: {args.jsonl} ({len(rows)} records)")
    print(f"Model: {args.model_name}")
    print(f"Top-k: {args.top_k}")
    print(f"Instruction prefix: {not args.raw_query}")

    for case_path in args.case_files:
        evaluate_case_file(
            case_path=case_path,
            rows=rows,
            chunks=chunks,
            encoder=encoder,
            index=index,
            top_k=args.top_k,
            use_instruction=not args.raw_query,
        )


if __name__ == "__main__":
    main()
