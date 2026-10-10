"""Tests for the training helpers in train_classic.py on tiny synthetic rows: no data files, no network."""

import unittest

from sklearn.naive_bayes import MultinomialNB

import train_classic
from slots import featurize, predict

ROWS = [
    {"id": "1", "text": "Taylor Swift tonight", "tokens": ["Taylor", "Swift", "tonight"],
     "tags": ["B-event_name", "I-event_name", "O"]},
    {"id": "2", "text": "Paris tomorrow", "tokens": ["Paris", "tomorrow"], "tags": ["B-city", "O"]},
]


class TypoCopyTest(unittest.TestCase):
    def test_copy_keeps_tags_and_marks_the_id(self):
        copy = train_classic.typo_copy(ROWS)
        self.assertEqual([r["id"] for r in copy], ["1:typo", "2:typo"])
        self.assertEqual([r["tags"] for r in copy], [r["tags"] for r in ROWS])
        self.assertEqual([len(r["tokens"]) for r in copy], [3, 2])

    def test_copy_is_reproducible(self):
        self.assertEqual(train_classic.typo_copy(ROWS), train_classic.typo_copy(ROWS))


class FitPredictTest(unittest.TestCase):
    def test_fit_then_predict_gives_one_tag_per_token(self):
        vec, model = train_classic.fit(MultinomialNB(alpha=0.1), featurize(ROWS), ROWS)
        predicted = predict(vec, model, featurize(ROWS))
        self.assertEqual([len(sentence) for sentence in predicted], [3, 2])

    def test_grids_cover_the_three_learners(self):
        self.assertEqual(set(train_classic.GRIDS), {"MultinomialNB", "LogisticRegression", "LinearSVC"})


if __name__ == "__main__":
    unittest.main()
