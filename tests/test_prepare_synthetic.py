"""Tests for prepare_synthetic.py: span location, BIO conversion and messy decontamination on tiny fixtures.

No data files, no network, no GPU.
"""

import unittest

import prepare_synthetic

MESSY = [
    {"id": "m1", "text": "Chi-Town events", "tokens": ["Chi", "-", "Town", "events"],
     "tags": ["B-event_name", "I-event_name", "I-event_name", "O"]},
]


class ToRowTest(unittest.TestCase):
    def test_span_becomes_bio_tags(self):
        row = prepare_synthetic.to_row("s:1", {"text": "Book a table in New York tonight",
                                               "spans": [["city", "New York"]]})
        self.assertEqual(row["tokens"], ["Book", "a", "table", "in", "New", "York", "tonight"])
        self.assertEqual(row["tags"], ["O", "O", "O", "O", "B-city", "I-city", "O"])
        self.assertEqual(row["id"], "s:1")

    def test_repeated_substring_is_found_after_the_previous_span(self):
        row = prepare_synthetic.to_row("s:2", {"text": "Paris to Paris tonight",
                                               "spans": [["city", "Paris"], ["city", "Paris"]]})
        self.assertEqual(row["tags"], ["B-city", "O", "B-city", "O"])

    def test_substring_inside_an_earlier_word_is_skipped(self):
        row = prepare_synthetic.to_row("s:5", {"text": "who is playing in la", "spans": [["city", "la"]]})
        self.assertEqual(row["tags"], ["O", "O", "O", "O", "B-city"])

    def test_missing_substring_fails_loudly(self):
        with self.assertRaises(ValueError):
            prepare_synthetic.to_row("s:3", {"text": "Jazz tonight", "spans": [["event_name", "Rock"]]})

    def test_substring_inside_a_word_fails_loudly(self):
        with self.assertRaises(ValueError):
            prepare_synthetic.to_row("s:4", {"text": "Lakers tonight", "spans": [["event_name", "ers"]]})


class DecontaminateTest(unittest.TestCase):
    def test_drops_rows_with_the_messy_event_name_tokens(self):
        rows = [
            {"id": "a", "text": "any chi-town events?", "tokens": ["any", "chi", "-", "town", "events", "?"]},
            {"id": "b", "text": "chi town events", "tokens": ["chi", "town", "events"]},
            {"id": "c", "text": "CHI - TOWN tonight", "tokens": ["CHI", "-", "TOWN", "tonight"]},
            {"id": "d", "text": "Chi-Town events", "tokens": ["Chi", "-", "Town", "events"]},
        ]
        kept, by_name, by_text = prepare_synthetic.decontaminate(rows, MESSY)
        self.assertEqual([r["id"] for r in kept], ["b"])
        self.assertEqual((by_name, by_text), (3, 0))

    def test_drops_rows_equal_to_a_messy_text(self):
        messy = [{"id": "m2", "text": "Good shows tonight", "tokens": ["Good", "shows", "tonight"], "tags": ["O"] * 3}]
        rows = [{"id": "x", "text": "good shows TONIGHT", "tokens": ["good", "shows", "TONIGHT"]}]
        kept, by_name, by_text = prepare_synthetic.decontaminate(rows, messy)
        self.assertEqual(kept, [])
        self.assertEqual((by_name, by_text), (0, 1))


if __name__ == "__main__":
    unittest.main()
