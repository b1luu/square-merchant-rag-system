from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluate_mosa_rag_jsonl import load_cases


class LoadCasesTests(unittest.TestCase):
    def _write_case_file(self, rows: list[dict]) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "cases.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_support_case_defaults_expected_outcome(self) -> None:
        path = self._write_case_file(
            [
                {
                    "id": "support_001",
                    "query": "what happens if i am sick?",
                    "expected_titles": ["Calling out sick and shift coverage"],
                }
            ]
        )

        cases = load_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].expected_outcome, "support")
        self.assertEqual(cases[0].expected_titles, ("Calling out sick and shift coverage",))

    def test_abstain_case_allows_missing_expected_targets(self) -> None:
        path = self._write_case_file(
            [
                {
                    "id": "abstain_001",
                    "query": "how do i reset the router?",
                    "expected_outcome": "abstain",
                }
            ]
        )

        cases = load_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].expected_outcome, "abstain")
        self.assertEqual(cases[0].expected_ids, ())
        self.assertEqual(cases[0].expected_titles, ())

    def test_invalid_expected_outcome_raises(self) -> None:
        path = self._write_case_file(
            [
                {
                    "id": "bad_001",
                    "query": "what is the policy?",
                    "expected_outcome": "maybe",
                    "expected_titles": ["Some title"],
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "invalid expected_outcome"):
            load_cases(path)

    def test_support_case_requires_expected_target(self) -> None:
        path = self._write_case_file(
            [
                {
                    "id": "bad_002",
                    "query": "what is the policy?",
                    "expected_outcome": "support",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "support case must include expected_ids or expected_titles"):
            load_cases(path)


if __name__ == "__main__":
    unittest.main()
