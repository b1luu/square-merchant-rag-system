from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mosa_rag.tracing import append_jsonl, build_answer_trace, load_jsonl, resolve_trace_path, summarize_traces


class TracingTests(unittest.TestCase):
    def test_trace_path_disabled_by_default(self) -> None:
        self.assertIsNone(resolve_trace_path({}))

    def test_append_and_load_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            append_jsonl(path, {"query": "hello", "answer_mode": "extractive"})

            rows = load_jsonl(path)

        self.assertEqual(rows, [{"query": "hello", "answer_mode": "extractive"}])

    def test_build_answer_trace_omits_answer_text(self) -> None:
        trace = build_answer_trace(
            route="/answer",
            query="what happens if i am sick?",
            top_k=2,
            raw_query=False,
            answer_mode="extractive",
            validation={"decision": "warn"},
            retrieval_confidence={"level": "medium", "score": 0.3888},
            abstained=False,
            verification={"unsupported_sentences": []},
            latency_ms=27.67,
            results=[{"id": "mosa_00127_policy_rule", "title": "Calling out sick and shift coverage"}],
        )

        self.assertNotIn("answer", trace)
        self.assertEqual(trace["validation_decision"], "warn")
        self.assertEqual(trace["result_ids"], ["mosa_00127_policy_rule"])

    def test_summarize_traces(self) -> None:
        summary = summarize_traces(
            [
                {
                    "query": "a",
                    "answer_mode": "extractive",
                    "validation_decision": "warn",
                    "abstained": False,
                    "latency_ms": 10,
                    "result_ids": ["one"],
                },
                {
                    "query": "b",
                    "answer_mode": "abstain",
                    "validation_decision": "abstain",
                    "abstained": True,
                    "latency_ms": 30,
                    "result_ids": ["two"],
                },
            ]
        )

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["abstention_rate"], 0.5)
        self.assertEqual(summary["warn_rate"], 0.5)
        self.assertEqual(summary["average_latency_ms"], 20.0)


if __name__ == "__main__":
    unittest.main()
