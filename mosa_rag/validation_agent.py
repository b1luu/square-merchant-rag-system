"""Deterministic validation agent for final answer gating."""

from __future__ import annotations

from dataclasses import dataclass

from mosa_rag.runtime import RetrievalConfidence
from mosa_rag.verification import AnswerVerification

AnswerMode = str
ValidationDecisionName = str


@dataclass(frozen=True)
class ValidationDecision:
    decision: ValidationDecisionName
    reasons: tuple[str, ...]
    answer_mode: AnswerMode
    confidence_level: str
    unsupported_sentences: tuple[str, ...]


def validate_answer(
    *,
    confidence: RetrievalConfidence,
    verification: AnswerVerification,
    answer_mode: AnswerMode,
    abstained: bool,
) -> ValidationDecision:
    """Combine retrieval confidence and grounding verification into a display decision."""
    reasons: list[str] = []

    if abstained or confidence.should_abstain or answer_mode == "abstain":
        reasons.append("retrieval confidence requires abstention")
        reasons.extend(confidence.reasons)
        return ValidationDecision(
            decision="abstain",
            reasons=tuple(dict.fromkeys(reasons)),
            answer_mode=answer_mode,
            confidence_level=confidence.level,
            unsupported_sentences=verification.unsupported_sentences,
        )

    if not verification.supported:
        reasons.append("answer contains details not supported by the retrieved records")
        reasons.extend(verification.unsupported_sentences)
        return ValidationDecision(
            decision="warn",
            reasons=tuple(dict.fromkeys(reasons)),
            answer_mode=answer_mode,
            confidence_level=confidence.level,
            unsupported_sentences=verification.unsupported_sentences,
        )

    if confidence.level == "medium":
        reasons.append("retrieval confidence is medium; answer should be reviewed with sources")
        return ValidationDecision(
            decision="warn",
            reasons=tuple(dict.fromkeys(reasons)),
            answer_mode=answer_mode,
            confidence_level=confidence.level,
            unsupported_sentences=(),
        )

    if answer_mode == "extractive":
        reasons.append("answer was assembled directly from retrieved records")

    if not reasons:
        reasons.append("answer passed retrieval confidence and grounding checks")

    return ValidationDecision(
        decision="show",
        reasons=tuple(dict.fromkeys(reasons)),
        answer_mode=answer_mode,
        confidence_level=confidence.level,
        unsupported_sentences=(),
    )
