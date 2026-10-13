"""Tests for evaluate.py: the compared pairs name scored models, and the test sets have the documented sizes.

No network and no GPU.
"""

import unittest

import evaluate


class ConfidenceSetupTest(unittest.TestCase):
    def test_every_pair_names_a_scored_model(self):
        names = {name for _, name in evaluate.CLASSIC + evaluate.TRANSFORMERS}
        for model, base in evaluate.PAIRS:
            self.assertIn(model, names)
            self.assertIn(base, names)
        for name in evaluate.CONFUSION:
            self.assertIn(name, names)

    def test_test_sets_have_the_readme_sizes(self):
        sizes = {test: len(rows) for test, rows in evaluate.test_sets().items()}
        self.assertEqual(sizes, {"SGD test": 784, "messy": 50, "codex_all": 260})


if __name__ == "__main__":
    unittest.main()
