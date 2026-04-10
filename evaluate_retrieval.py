from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from retrieve_pdf import build_faiss_index, read_pdf_chunks, retrieve

DEFAULT_PDF = Path("data/Mosa Employee Handbook.pdf")


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_page: int
    expected_substrings: tuple[str, ...]


EVAL_CASES = (
    EvalCase(
        question="How many discounted drinks per month do employees receive?",
        expected_page=10,
        expected_substrings=("10 drinks per month", "10% discount"),
    ),
    EvalCase(
        question="How early can employees clock in before their scheduled shift?",
        expected_page=2,
        expected_substrings=("5 minutes before your scheduled shift time",),
    ),
    EvalCase(
        question="What breaks are employees entitled to?",
        expected_page=9,
        expected_substrings=("30 minute meal break", "10 minute rest break"),
    ),
    EvalCase(
        question="What are the front of house responsibilities?",
        expected_page=1,
        expected_substrings=("Front of House:", "Open & close bar and front"),
    ),
    EvalCase(
        question="Where can employees find their POS passcode?",
        expected_page=5,
        expected_substrings=("Your POS passcode is on the bottom",),
    ),
    EvalCase(
        question="What should employees do if they are more than 5 minutes late?",
        expected_page=9,
        expected_substrings=("notify the person you are working with",),
    ),
    EvalCase(
        question="What is the policy on outside food or drinks in the kitchen or bar areas?",
        expected_page=8,
        expected_substrings=("No outside food or drinks are allowed in the kitchen or bar areas",),
    ),
    EvalCase(
        question="Where should personal belongings be stored?",
        expected_page=10,
        expected_substrings=("stored in lockers located in the back storage",),
    ),
)


def chunk_matches(case: EvalCase, page_number: int, text: str) -> bool:
    normalized_text = text.lower()
    return page_number == case.expected_page and all(
        substring.lower() in normalized_text for substring in case.expected_substrings
    )


def evaluate_setting(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    use_instruction: bool,
) -> dict[str, float | int]:
    chunks = read_pdf_chunks(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    encoder, index, _ = build_faiss_index(chunks=chunks, model_name="BAAI/bge-base-en-v1.5")

    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_rank_total = 0.0

    for case in EVAL_CASES:
        results = retrieve(
            encoder=encoder,
            index=index,
            chunks=chunks,
            query=case.question,
            top_k=top_k,
            use_instruction=use_instruction,
        )

        match_rank = None
        for rank, (_, chunk) in enumerate(results, start=1):
            if chunk_matches(case, chunk.page_number, chunk.text):
                match_rank = rank
                break

        if match_rank == 1:
            hit_at_1 += 1
        if match_rank is not None:
            hit_at_k += 1
            reciprocal_rank_total += 1.0 / match_rank

    case_count = len(EVAL_CASES)
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks": len(chunks),
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "mrr": reciprocal_rank_total / case_count,
        "instruction": int(use_instruction),
    }


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval settings against a small handbook question set."
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=DEFAULT_PDF,
        help=f"PDF to evaluate (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="60,80,100,120",
        help="Comma-separated chunk sizes to evaluate",
    )
    parser.add_argument(
        "--chunk-overlaps",
        default="10,15,20,30",
        help="Comma-separated chunk overlaps to evaluate",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k retrieval depth for evaluation (default: 3)",
    )
    parser.add_argument(
        "--no-instruction",
        action="store_true",
        help="Do not prepend the BGE retrieval instruction to questions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunk_sizes = parse_int_list(args.chunk_sizes)
    chunk_overlaps = parse_int_list(args.chunk_overlaps)
    use_instruction = not args.no_instruction

    results = []
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            if chunk_overlap >= chunk_size:
                continue
            results.append(
                evaluate_setting(
                    pdf_path=args.pdf_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=args.top_k,
                    use_instruction=use_instruction,
                )
            )

    results.sort(key=lambda row: (row["hit_at_1"], row["hit_at_k"], row["mrr"]), reverse=True)

    print(f"PDF: {args.pdf_path}")
    print(f"Cases: {len(EVAL_CASES)}")
    print(f"Top-k: {args.top_k}")
    print(f"Instruction: {use_instruction}")
    print()
    print("chunk_size chunk_overlap chunks hit@1 hit@k mrr")
    for row in results:
        print(
            f"{row['chunk_size']:>10} {row['chunk_overlap']:>13} {row['chunks']:>6} "
            f"{row['hit_at_1']:>5}/{len(EVAL_CASES)} {row['hit_at_k']:>5}/{len(EVAL_CASES)} "
            f"{row['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()
