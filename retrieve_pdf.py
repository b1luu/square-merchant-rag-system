from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from pypdf import PdfReader
from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEFAULT_ONNX_FILE = "onnx/model.onnx"


@dataclass
class Chunk:
    chunk_id: int
    page_number: int
    text: str


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def read_pdf_chunks(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int = 40,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    reader = PdfReader(str(pdf_path))
    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap

    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_whitespace(page.extract_text() or "")
        if not text:
            continue

        words = text.split()
        if not words:
            continue

        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()

            if len(chunk_text) >= min_chunk_chars:
                # Retrieval quality depends heavily on chunk granularity:
                # chunks that are too large mix unrelated facts together.
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        page_number=page_number,
                        text=chunk_text,
                    )
                )

            if end == len(words):
                break

    if not chunks:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "If this is a scanned PDF, you need OCR before retrieval."
        )

    return chunks


class BGEOnnxEncoder:
    def __init__(self, model_name: str, onnx_file: str = DEFAULT_ONNX_FILE) -> None:
        # Pull only the files needed for tokenization + ONNX inference and let
        # the Hugging Face cache handle reuse across runs.
        local_dir = snapshot_download(model_name, allow_patterns=[onnx_file, "*.json", "*.txt", "vocab.txt"])
        onnx_path = Path(local_dir) / onnx_file
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def encode(self, texts: list[str], batch_size: int = 32, max_length: int = 512) -> np.ndarray:
        embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )

            ort_inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in encoded:
                ort_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

            last_hidden_state = self.session.run(None, ort_inputs)[0]
            # The BGE model card's reference code uses CLS pooling, so we take
            # the embedding at position 0 for each sequence.
            cls_embeddings = last_hidden_state[:, 0, :]
            # Normalize so FAISS inner-product search behaves like cosine search.
            norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            embeddings.append((cls_embeddings / norms).astype(np.float32))

        return np.vstack(embeddings)


def build_faiss_index(
    chunks: list[Chunk],
    model_name: str,
) -> tuple[BGEOnnxEncoder, faiss.IndexFlatIP, np.ndarray]:
    encoder = BGEOnnxEncoder(model_name)
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = encoder.encode(chunk_texts, batch_size=32)

    # IndexFlatIP is exact inner-product search. With normalized vectors, this
    # is a simple way to rank by cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return encoder, index, embeddings


def format_query(query: str, use_instruction: bool) -> str:
    # BGE recommends adding this instruction to retrieval queries, but not to
    # the document passages.
    return f"{QUERY_INSTRUCTION}{query}" if use_instruction else query


def retrieve(
    encoder: BGEOnnxEncoder,
    index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    query: str,
    top_k: int,
    use_instruction: bool,
) -> list[tuple[float, Chunk]]:
    query_embedding = encoder.encode([format_query(query, use_instruction)])

    top_k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, top_k)

    results: list[tuple[float, Chunk]] = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        results.append((float(score), chunks[int(idx)]))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve the most relevant chunks from a PDF with BGE ONNX + FAISS."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument("query", help="Question to search for in the PDF")
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=f"Embedding model to use (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Chunk size in words (default: 250)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in words (default: 50)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many matching chunks to print (default: 3)",
    )
    parser.add_argument(
        "--raw-query",
        action="store_true",
        help="Do not prepend the BGE retrieval instruction to the query",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf_path}")

    chunks = read_pdf_chunks(
        pdf_path=args.pdf_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Loaded {len(chunks)} chunks from {args.pdf_path}")
    print(f"Embedding with {args.model_name} using direct ONNX runtime")

    encoder, index, _ = build_faiss_index(chunks=chunks, model_name=args.model_name)
    results = retrieve(
        encoder=encoder,
        index=index,
        chunks=chunks,
        query=args.query,
        top_k=args.top_k,
        use_instruction=not args.raw_query,
    )

    print(f"\nQuery: {args.query}")
    print(f"Top {len(results)} matches:")

    for rank, (score, chunk) in enumerate(results, start=1):
        print("\n" + "=" * 100)
        print(
            f"Rank {rank} | score={score:.4f} | page={chunk.page_number} | chunk_id={chunk.chunk_id}"
        )
        print("-" * 100)
        print(textwrap.fill(chunk.text, width=100))


if __name__ == "__main__":
    main()
