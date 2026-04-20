from __future__ import annotations

import unittest

from mosa_rag.verification import verify_answer_support


class AnswerVerificationTests(unittest.TestCase):
    def test_supported_answer_passes(self) -> None:
        verification = verify_answer_support(
            "If you are sick, contact management early so they can handle shift coverage. "
            "Sources: Calling out sick and shift coverage",
            [
                "Calling out sick and shift coverage. If you are sick and need to call out, "
                "contact management as early as possible. Management will handle finding shift coverage."
            ],
        )

        self.assertTrue(verification.supported)
        self.assertEqual(verification.unsupported_sentences, ())
        self.assertGreaterEqual(verification.coverage_score, 0.55)

    def test_unsupported_sentence_is_flagged(self) -> None:
        verification = verify_answer_support(
            "If you are sick, contact management early. "
            "Employees also receive a free taxi ride home after calling out.",
            [
                "Calling out sick and shift coverage. If you are sick and need to call out, "
                "contact management as early as possible. Management will handle finding shift coverage."
            ],
        )

        self.assertFalse(verification.supported)
        self.assertEqual(
            verification.unsupported_sentences,
            ("Employees also receive a free taxi ride home after calling out.",),
        )

    def test_sources_line_is_not_checked_as_answer_content(self) -> None:
        verification = verify_answer_support(
            "Sources: Calling out sick and shift coverage",
            ["Calling out sick and shift coverage."],
        )

        self.assertTrue(verification.supported)
        self.assertEqual(verification.checked_sentences, 0)


if __name__ == "__main__":
    unittest.main()
