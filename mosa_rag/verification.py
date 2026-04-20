"""Deterministic answer-grounding checks over retrieved evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
        "you",
        "your",
    }
)


@dataclass(frozen=True)
class AnswerVerification:
    supported: bool
    coverage_score: float
    checked_sentences: int
    unsupported_sentences: tuple[str, ...]


def _normalize_token(token: str) -> str:
    text = token.lower()
    if text.endswith("ies") and len(text) > 4:
        return text[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if text.endswith(suffix) and len(text) - len(suffix) >= 4:
            return text[: -len(suffix)]
    return text


def _content_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall(text.lower()):
        token = _normalize_token(raw)
        if len(token) <= 2 or token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _sentences(answer: str) -> list[str]:
    sentences: list[str] = []
    for raw in _SENTENCE_RE.findall(answer):
        sentence = raw.strip()
        if not sentence:
            continue
        if sentence.lower().startswith("sources:"):
            continue
        sentences.append(sentence)
    return sentences


def verify_answer_support(
    answer: str,
    evidence_texts: list[str],
    *,
    min_sentence_coverage: float = 0.55,
    min_supported_terms: int = 2,
) -> AnswerVerification:
    """Flag answer sentences whose content words are poorly supported by retrieved evidence."""
    evidence_tokens: set[str] = set()
    for text in evidence_texts:
        evidence_tokens.update(_content_tokens(text))

    unsupported: list[str] = []
    scores: list[float] = []
    for sentence in _sentences(answer):
        sentence_tokens = _content_tokens(sentence)
        if not sentence_tokens:
            continue

        supported_terms = sentence_tokens & evidence_tokens
        coverage = len(supported_terms) / len(sentence_tokens)
        scores.append(coverage)
        if coverage < min_sentence_coverage or len(supported_terms) < min_supported_terms:
            unsupported.append(sentence)

    if not scores:
        return AnswerVerification(
            supported=True,
            coverage_score=1.0,
            checked_sentences=0,
            unsupported_sentences=(),
        )

    return AnswerVerification(
        supported=not unsupported,
        coverage_score=round(sum(scores) / len(scores), 4),
        checked_sentences=len(scores),
        unsupported_sentences=tuple(unsupported),
    )
