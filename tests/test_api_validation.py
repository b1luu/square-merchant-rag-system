from __future__ import annotations

import unittest

from serve_mosa_rag import MAX_QUERY_CHARS, parse_request_options


class ApiValidationTests(unittest.TestCase):
    def _parse(self, payload: dict):
        return parse_request_options(payload, top_k_default=2, raw_query_default=False)

    def test_valid_payload(self) -> None:
        options = self._parse(
            {
                "query": " what happens if i am sick? ",
                "top_k": 3,
                "raw_query": True,
                "show_context": False,
                "stream": False,
                "allow_low_confidence": True,
            }
        )

        self.assertEqual(options.query, "what happens if i am sick?")
        self.assertEqual(options.top_k, 3)
        self.assertTrue(options.raw_query)
        self.assertTrue(options.allow_low_confidence)

    def test_top_k_must_be_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k must be an integer"):
            self._parse({"query": "hello", "top_k": "abc"})

    def test_top_k_must_be_in_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k must be an integer between 1 and 20"):
            self._parse({"query": "hello", "top_k": 0})

        with self.assertRaisesRegex(ValueError, "top_k must be an integer between 1 and 20"):
            self._parse({"query": "hello", "top_k": 21})

    def test_query_must_be_non_empty_string(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty 'query'"):
            self._parse({"query": "   "})

        with self.assertRaisesRegex(ValueError, "query must be a string"):
            self._parse({"query": 123})

    def test_query_length_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "characters or fewer"):
            self._parse({"query": "x" * (MAX_QUERY_CHARS + 1)})

    def test_boolean_fields_must_be_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "stream must be a boolean"):
            self._parse({"query": "hello", "stream": "false"})


if __name__ == "__main__":
    unittest.main()
