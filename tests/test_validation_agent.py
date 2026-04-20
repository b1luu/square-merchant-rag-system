from __future__ import annotations

import unittest

from mosa_rag.runtime import RetrievalConfidence
from mosa_rag.validation_agent import validate_answer
from mosa_rag.verification import AnswerVerification


def _confidence(level: str = "high", should_abstain: bool = False) -> RetrievalConfidence:
    return RetrievalConfidence(
        score=0.8 if level == "high" else 0.4,
        level=level,
        should_abstain=should_abstain,
        top_score=0.8,
        score_margin=0.2,
        supporting_hits=1,
        anchor_ratio=1.0,
        reasons=("low confidence reason",) if should_abstain else (),
    )


def _verification(supported: bool = True) -> AnswerVerification:
    return AnswerVerification(
        supported=supported,
        coverage_score=1.0 if supported else 0.3,
        checked_sentences=1,
        unsupported_sentences=() if supported else ("Unsupported sentence.",),
    )


class ValidationAgentTests(unittest.TestCase):
    def test_abstains_when_retrieval_requires_abstention(self) -> None:
        decision = validate_answer(
            confidence=_confidence(level="low", should_abstain=True),
            verification=_verification(),
            answer_mode="abstain",
            abstained=True,
        )

        self.assertEqual(decision.decision, "abstain")
        self.assertIn("retrieval confidence requires abstention", decision.reasons)
        self.assertIn("low confidence reason", decision.reasons)

    def test_warns_on_unsupported_details(self) -> None:
        decision = validate_answer(
            confidence=_confidence(),
            verification=_verification(supported=False),
            answer_mode="llm",
            abstained=False,
        )

        self.assertEqual(decision.decision, "warn")
        self.assertEqual(decision.unsupported_sentences, ("Unsupported sentence.",))

    def test_warns_on_medium_confidence(self) -> None:
        decision = validate_answer(
            confidence=_confidence(level="medium"),
            verification=_verification(),
            answer_mode="extractive",
            abstained=False,
        )

        self.assertEqual(decision.decision, "warn")
        self.assertIn("retrieval confidence is medium", decision.reasons[0])

    def test_shows_high_confidence_verified_answer(self) -> None:
        decision = validate_answer(
            confidence=_confidence(level="high"),
            verification=_verification(),
            answer_mode="llm",
            abstained=False,
        )

        self.assertEqual(decision.decision, "show")
        self.assertEqual(decision.answer_mode, "llm")
        self.assertEqual(decision.unsupported_sentences, ())


if __name__ == "__main__":
    unittest.main()
