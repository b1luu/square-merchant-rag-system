"""Resident retrieval runtime shared by CLI and warm local API."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from mosa_rag.faiss_cache import default_cache_root, load_or_build_faiss_index
from mosa_rag.retrieve_jsonl import build_chunks, load_rows
from mosa_rag.verification import AnswerVerification, verify_answer_support
from retrieve_pdf import MODEL_NAME, Chunk, retrieve

ANSWER_TOP_K_DEFAULT = 2
CONFIDENCE_SUPPORT_WINDOW = 0.05
CONFIDENCE_LOW_THRESHOLD = 0.22
CONFIDENCE_HIGH_THRESHOLD = 0.5
CONFIDENCE_MIN_TOP_SCORE = 0.52
CONFIDENCE_MIN_TOP_SCORE_WITHOUT_ANCHOR = 0.56
CONFIDENCE_MIN_MARGIN_WITHOUT_ANCHOR = 0.03
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "do",
        "does",
        "for",
        "get",
        "how",
        "i",
        "if",
        "im",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "we",
        "what",
        "when",
        "where",
        "who",
        "why",
        "you",
        "your",
    }
)

INSTRUCTIONS = (
    "You are answering internal Mosa Tea staff questions. "
    "Only use the provided retrieved records—treat them as the only source of truth. "
    "If the records are insufficient or conflicting, say so plainly. "
    "Never present general knowledge or guesses as if they came from the records."
)


def _append_field(lines: list[str], field: str, value: object) -> None:
    if not value:
        return
    if isinstance(value, list):
        joined = "; ".join(str(item) for item in value)
        if joined:
            lines.append(f"{field}: {joined}")
        return
    text = str(value).strip()
    if text:
        lines.append(f"{field}: {text}")


def format_record(record: dict, score: float, *, compact: bool) -> str:
    lines = [
        f"title: {record.get('title', '')}",
        f"type: {record.get('type', '')}",
        f"source: {record.get('source_file', '')} page {record.get('source_page', '')}",
        f"score: {score:.4f}",
    ]

    structured_fields = ("rules", "steps", "ingredients", "storage_life", "threshold", "action")
    has_structured_payload = any(record.get(field) for field in structured_fields)
    if compact:
        for field in structured_fields:
            _append_field(lines, field, record.get(field))
        if not has_structured_payload:
            _append_field(lines, "retrieval_text", record.get("retrieval_text"))
    else:
        _append_field(lines, "id", record.get("id"))
        for field in ("retrieval_text", *structured_fields):
            _append_field(lines, field, record.get(field))
    return "\n".join(lines)


def build_context(rows: list[dict], results: list[tuple[float, Chunk]], *, compact: bool) -> str:
    blocks: list[str] = []
    for rank, (score, chunk) in enumerate(results, start=1):
        record = rows[chunk.chunk_id]
        blocks.append(f"[record {rank}]\n{format_record(record, score, compact=compact)}")
    return "\n\n".join(blocks)


def _normalize_range(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    if value <= lower:
        return 0.0
    if value >= upper:
        return 1.0
    return (value - lower) / (upper - lower)


def _normalize_token(token: str) -> str:
    text = token.lower()
    if text.endswith("ies") and len(text) > 4:
        return text[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if text.endswith(suffix) and len(text) - len(suffix) >= 4:
            return text[: -len(suffix)]
    return text


def _important_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(query.lower()):
        token = _normalize_token(raw)
        if len(token) <= 2 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _tokens_for_text(text: str) -> list[str]:
    return [_normalize_token(token) for token in _TOKEN_RE.findall(text.lower())]


def _tokens_match(query_term: str, text_token: str) -> bool:
    if query_term == text_token:
        return True
    if len(query_term) >= 4 and query_term in text_token:
        return True
    if len(text_token) >= 4 and text_token in query_term:
        return True
    return False


def _anchor_ratio(query_terms: list[str], candidate_texts: list[str]) -> float:
    if not query_terms:
        return 0.0

    candidate_tokens: list[str] = []
    for text in candidate_texts:
        candidate_tokens.extend(_tokens_for_text(text))
    if not candidate_tokens:
        return 0.0

    hits = 0
    for term in query_terms:
        if any(_tokens_match(term, token) for token in candidate_tokens):
            hits += 1
    return hits / len(query_terms)


def _record_confidence_text(record: dict) -> str:
    parts = [
        str(record.get("title", "")),
        str(record.get("type", "")),
        str(record.get("retrieval_text", "")),
    ]
    return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class RetrievalConfidence:
    score: float
    level: str
    should_abstain: bool
    top_score: float
    score_margin: float
    supporting_hits: int
    anchor_ratio: float
    reasons: tuple[str, ...]


def assess_retrieval_confidence(
    query: str,
    rows: list[dict],
    results: list[tuple[float, Chunk]],
) -> RetrievalConfidence:
    if not results:
        return RetrievalConfidence(
            score=0.0,
            level="low",
            should_abstain=True,
            top_score=0.0,
            score_margin=0.0,
            supporting_hits=0,
            anchor_ratio=0.0,
            reasons=("no retrieval results were returned",),
        )

    scores = [score for score, _ in results]
    top_score = scores[0]
    score_margin = top_score - scores[1] if len(scores) > 1 else top_score
    support_window = top_score - CONFIDENCE_SUPPORT_WINDOW
    supporting_hits = sum(1 for score in scores[:3] if score >= support_window)

    query_terms = _important_terms(query)
    top_records = [rows[chunk.chunk_id] for _, chunk in results[:3]]
    top1_anchor = _anchor_ratio(query_terms, [_record_confidence_text(top_records[0])])
    context_anchor = _anchor_ratio(query_terms, [_record_confidence_text(record) for record in top_records])
    anchor_ratio = (0.7 * top1_anchor) + (0.3 * context_anchor)

    normalized_top = _normalize_range(top_score, 0.48, 0.82)
    normalized_margin = _normalize_range(score_margin, 0.0, 0.12)
    normalized_support = supporting_hits / max(1, min(3, len(results)))
    score = (
        (0.5 * normalized_top)
        + (0.15 * normalized_margin)
        + (0.25 * anchor_ratio)
        + (0.10 * normalized_support)
    )

    reasons: list[str] = []
    should_abstain = False
    if top_score < CONFIDENCE_MIN_TOP_SCORE and anchor_ratio < 0.2:
        should_abstain = True
        reasons.append("top match is weak and query terms barely appear in the retrieved evidence")
    if (
        top_score < CONFIDENCE_MIN_TOP_SCORE_WITHOUT_ANCHOR
        and anchor_ratio == 0.0
        and score_margin < CONFIDENCE_MIN_MARGIN_WITHOUT_ANCHOR
    ):
        should_abstain = True
        reasons.append("top matches cluster together without any lexical anchor to the question")
    if score < CONFIDENCE_LOW_THRESHOLD:
        should_abstain = True
        reasons.append("overall retrieval confidence is below the abstain threshold")

    if should_abstain:
        level = "low"
    elif score >= CONFIDENCE_HIGH_THRESHOLD:
        level = "high"
    else:
        level = "medium"
        if anchor_ratio < 0.25:
            reasons.append("retrieval is plausible but only weakly anchored to the query wording")
        if top_score < 0.6:
            reasons.append("top similarity score is modest, so answer carefully")

    return RetrievalConfidence(
        score=round(score, 4),
        level=level,
        should_abstain=should_abstain,
        top_score=round(top_score, 4),
        score_margin=round(score_margin, 4),
        supporting_hits=supporting_hits,
        anchor_ratio=round(anchor_ratio, 4),
        reasons=tuple(dict.fromkeys(reasons)),
    )


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

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = ANSWER_TOP_K_DEFAULT,
        raw_query: bool = False,
    ) -> list[tuple[float, Chunk]]:
        with self._retrieve_lock:
            return retrieve(
                encoder=self.encoder,
                index=self.index,
                chunks=self.chunks,
                query=query,
                top_k=min(top_k, len(self.chunks)),
                use_instruction=not raw_query,
            )

    def assess_confidence(
        self,
        query: str,
        results: list[tuple[float, Chunk]],
    ) -> RetrievalConfidence:
        return assess_retrieval_confidence(query, self.rows, results)

    def retrieve_with_confidence(
        self,
        query: str,
        *,
        top_k: int = ANSWER_TOP_K_DEFAULT,
        raw_query: bool = False,
    ) -> tuple[list[tuple[float, Chunk]], RetrievalConfidence]:
        results = self.retrieve(query, top_k=top_k, raw_query=raw_query)
        return results, self.assess_confidence(query, results)

    def build_prompt_bundle(
        self,
        query: str,
        *,
        top_k: int = ANSWER_TOP_K_DEFAULT,
        raw_query: bool = False,
    ) -> tuple[str, str, str, list[tuple[float, Chunk]], RetrievalConfidence]:
        results, confidence = self.retrieve_with_confidence(query, top_k=top_k, raw_query=raw_query)
        prompt_context = build_context(self.rows, results, compact=True)
        display_context = build_context(self.rows, results, compact=False)
        return build_prompt(query, prompt_context), prompt_context, display_context, results, confidence

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

    def verify_answer(
        self,
        answer: str,
        results: list[tuple[float, Chunk]],
    ) -> AnswerVerification:
        evidence_texts: list[str] = []
        for _, chunk in results:
            record = self.rows[chunk.chunk_id]
            evidence_texts.extend(
                str(record.get(field, ""))
                for field in ("title", "type", "retrieval_text", "rules", "steps", "ingredients", "action")
                if record.get(field)
            )
        return verify_answer_support(answer, evidence_texts)
