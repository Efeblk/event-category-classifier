"""Tests for confidence.py: the compared pairs name scored models, and the test sets have the documented sizes.

No network and no GPU.
"""

import unittest

import confidence


class ConfidenceSetupTest(unittest.TestCase):
    def test_every_pair_names_a_scored_model(self):
        names = {name for _, name in confidence.CLASSIC + confidence.TRANSFORMERS}
        for model, base in confidence.PAIRS:
            self.assertIn(model, names)
            self.assertIn(base, names)

    def test_test_sets_have_the_readme_sizes(self):
        sizes = {test: len(rows) for test, rows in confidence.test_sets().items()}
        self.assertEqual(sizes, {"SGD test": 784, "messy": 50, "codex_all": 260})


if __name__ == "__main__":
    unittest.main()
