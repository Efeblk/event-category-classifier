import json
import unittest
from collections import defaultdict
from pathlib import Path

from prepare_requests import build_rows, normalize


class RequestCorpusTests(unittest.TestCase):
    def test_bilingual_paraphrases_keep_one_family(self):
        rows, audit = build_rows()
        groups = defaultdict(list)
        for row in rows:
            groups[row["group_id"]].append(row)
        self.assertEqual(set(audit["languages"]), {"Turkish", "English"})
        self.assertGreater(audit["language_counts"]["Turkish"], 0)
        self.assertGreater(audit["language_counts"]["English"], 0)
        for group in groups.values():
            self.assertEqual(len({row["label"] for row in group}), 1)
            self.assertGreaterEqual(len(group), 2)

    def test_synthetic_corpus_stays_below_course_admission(self):
        rows, audit = build_rows()
        self.assertLess(len(rows), 1000)
        self.assertIn("synthetic", audit["provenance"])
        self.assertIn("Does not satisfy", audit["course_requirement_status"])
        self.assertEqual(len({normalize(row["text"]) for row in rows}), len(rows))

    def test_challenge_has_no_exact_training_text_overlap(self):
        rows, _ = build_rows()
        challenge_path = Path(__file__).resolve().parents[1] / "data/request_challenge.json"
        challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
        training = {normalize(row["text"]) for row in rows}
        held_out = {normalize(row["text"]) for row in challenge["records"]}
        self.assertFalse(training & held_out)
        self.assertIn("AI-authored", challenge["provenance"])


if __name__ == "__main__":
    unittest.main()
