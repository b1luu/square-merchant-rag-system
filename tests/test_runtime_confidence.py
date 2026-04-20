from __future__ import annotations

import unittest

from mosa_rag.runtime import assess_retrieval_confidence
from retrieve_pdf import Chunk


def _chunk(chunk_id: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_name="test.jsonl",
        source_path="/tmp/test.jsonl",
        page_number=1,
        text="",
    )


class RetrievalConfidenceTests(unittest.TestCase):
    def test_no_results_abstains(self) -> None:
        confidence = assess_retrieval_confidence("how do i reset the router?", [], [])

        self.assertTrue(confidence.should_abstain)
        self.assertEqual(confidence.level, "low")
        self.assertEqual(confidence.reasons, ("no retrieval results were returned",))

    def test_weak_unanchored_results_abstain(self) -> None:
        rows = [
            {
                "title": "Square POS clock in and breaks",
                "type": "pos_procedure",
                "retrieval_text": "Use the Square tablet to clock in, clock out, and track breaks.",
            },
            {
                "title": "Hungry Panda order workflow",
                "type": "pos_procedure",
                "retrieval_text": "Mirror Hungry Panda orders in the POS and print tickets.",
            },
        ]
        results = [(0.51, _chunk(0)), (0.50, _chunk(1))]

        confidence = assess_retrieval_confidence("how do i re-pair the card reader?", rows, results)

        self.assertTrue(confidence.should_abstain)
        self.assertEqual(confidence.level, "low")
        self.assertEqual(confidence.anchor_ratio, 0.0)
        self.assertIn("top match is weak", "; ".join(confidence.reasons))

    def test_strong_anchored_results_do_not_abstain(self) -> None:
        rows = [
            {
                "title": "Calling out sick and shift coverage",
                "type": "policy_rule",
                "retrieval_text": (
                    "If you are sick and need to call out, contact management as early as possible. "
                    "Management will handle finding shift coverage."
                ),
            },
            {
                "title": "Hygiene illness policy",
                "type": "policy_rule",
                "retrieval_text": "Do not work sick with contagious illnesses; notify management.",
            },
        ]
        results = [(0.78, _chunk(0)), (0.62, _chunk(1))]

        confidence = assess_retrieval_confidence("what happens if i am sick?", rows, results)

        self.assertFalse(confidence.should_abstain)
        self.assertEqual(confidence.level, "high")
        self.assertGreaterEqual(confidence.anchor_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
