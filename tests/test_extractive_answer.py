from __future__ import annotations

import unittest

from mosa_rag.runtime import build_extractive_answer
from retrieve_pdf import Chunk


def _chunk(chunk_id: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_name="test.jsonl",
        source_path="/tmp/test.jsonl",
        page_number=1,
        text="",
    )


class ExtractiveAnswerTests(unittest.TestCase):
    def test_builds_answer_from_top_records(self) -> None:
        rows = [
            {
                "title": "Calling out sick and shift coverage",
                "retrieval_text": "If you are sick, contact management as early as possible.",
            },
            {
                "title": "Hygiene illness policy",
                "retrieval_text": "Do not work sick with contagious illnesses; notify management.",
            },
        ]

        answer = build_extractive_answer(rows, [(0.8, _chunk(0)), (0.7, _chunk(1))])

        self.assertIn("Calling out sick and shift coverage:", answer)
        self.assertIn("If you are sick, contact management as early as possible.", answer)
        self.assertIn("Sources: Calling out sick and shift coverage, Hygiene illness policy", answer)

    def test_uses_structured_fields_before_retrieval_text(self) -> None:
        rows = [
            {
                "title": "Boba prep",
                "retrieval_text": "Generic boba retrieval text.",
                "steps": ["Cook boba.", "Rest boba."],
            }
        ]

        answer = build_extractive_answer(rows, [(0.8, _chunk(0))])

        self.assertIn("Cook boba. Rest boba.", answer)
        self.assertNotIn("Generic boba retrieval text.", answer)

    def test_empty_results_abstain_text(self) -> None:
        answer = build_extractive_answer([], [])

        self.assertEqual(answer, "The retrieved records do not clearly answer this question.")


if __name__ == "__main__":
    unittest.main()
