"""Resident retrieval runtime shared by CLI and warm local API."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from mosa_rag.faiss_cache import default_cache_root, load_or_build_faiss_index
from mosa_rag.retrieve_jsonl import build_chunks, load_rows
from retrieve_pdf import MODEL_NAME, Chunk, retrieve

INSTRUCTIONS = (
    "You are answering internal Mosa Tea staff questions. "
    "Only use the provided retrieved records—treat them as the only source of truth. "
    "If the records are insufficient or conflicting, say so plainly. "
    "Never present general knowledge or guesses as if they came from the records."
)


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
            lines.append(f"{field}: {'; '.join(str(item) for item in value)}")
        else:
            lines.append(f"{field}: {value}")
    return "\n".join(lines)


def build_context(rows: list[dict], results: list[tuple[float, Chunk]]) -> str:
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


def build_prompt(query: str, context: str) -> str:
    return f"{INSTRUCTIONS}\n\n{build_user_input(query, context)}"


def resolve_cache_root(*, jsonl: Path, cache_dir: Path | None, no_cache: bool) -> Path | None:
    if no_cache:
        return None
    if cache_dir is not None:
        return cache_dir
    env = os.getenv("BGE_RAG_CACHE_DIR")
    if env:
        return Path(env)
    return default_cache_root(jsonl)


@dataclass(frozen=True)
class RetrievedRecord:
    rank: int
    score: float
    id: str
    title: str
    type: str
    source_file: str
    source_page: int
    retrieval_text: str


class ResidentRetriever:
    """Keeps corpus, encoder, and FAISS index resident in memory for repeated queries."""

    def __init__(
        self,
        *,
        jsonl: Path = Path("normalized_mosa_rag.jsonl"),
        model_name: str = MODEL_NAME,
        cache_dir: Path | None = None,
        no_cache: bool = False,
    ) -> None:
        self.jsonl = jsonl
        self.model_name = model_name
        self.rows = load_rows(jsonl)
        self.chunks = build_chunks(self.rows, jsonl)
        cache_root = resolve_cache_root(jsonl=jsonl, cache_dir=cache_dir, no_cache=no_cache)
        self.encoder, self.index, _ = load_or_build_faiss_index(
            chunks=self.chunks,
            model_name=model_name,
            source_jsonl=jsonl,
            cache_root=cache_root,
            no_cache=no_cache,
        )
        self._retrieve_lock = threading.Lock()

    def retrieve(self, query: str, *, top_k: int = 5, raw_query: bool = False) -> list[tuple[float, Chunk]]:
        with self._retrieve_lock:
            return retrieve(
                encoder=self.encoder,
                index=self.index,
                chunks=self.chunks,
                query=query,
                top_k=min(top_k, len(self.chunks)),
                use_instruction=not raw_query,
            )

    def build_prompt_bundle(
        self,
        query: str,
        *,
        top_k: int = 5,
        raw_query: bool = False,
    ) -> tuple[str, str, list[tuple[float, Chunk]]]:
        results = self.retrieve(query, top_k=top_k, raw_query=raw_query)
        context = build_context(self.rows, results)
        return build_prompt(query, context), context, results

    def serialize_results(self, results: list[tuple[float, Chunk]]) -> list[RetrievedRecord]:
        payload: list[RetrievedRecord] = []
        for rank, (score, chunk) in enumerate(results, start=1):
            record = self.rows[chunk.chunk_id]
            payload.append(
                RetrievedRecord(
                    rank=rank,
                    score=score,
                    id=str(record.get("id", "")),
                    title=str(record.get("title", "")),
                    type=str(record.get("type", "")),
                    source_file=str(record.get("source_file", "")),
                    source_page=int(record.get("source_page") or 0),
                    retrieval_text=str(record.get("retrieval_text", "")),
                )
            )
        return payload
