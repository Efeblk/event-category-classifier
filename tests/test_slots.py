"""Tests for slots.py: BIO conversion and cleaning, span scoring, typo noise, features and the dictionary tagger.

Standard library plus scikit-learn only, no network and no GPU.
"""

import random
import unittest

from sklearn.svm import LinearSVC

from slots import (DictionaryTagger, add_typos, bio_tags, clean_split, dedupe, drop_leaks, featurize,
                   fit_vectorizer, predict, repair_bio, slot_counts, span_scores, spans, token_features)


class BioTagsTest(unittest.TestCase):
    def test_single_token_span(self):
        tokens, tags = bio_tags("Book a table in Paris tonight", [(16, 21, "city")])
        self.assertEqual(tokens, ["Book", "a", "table", "in", "Paris", "tonight"])
        self.assertEqual(tags, ["O", "O", "O", "O", "B-city", "O"])

    def test_multi_token_span_gets_begin_then_inside(self):
        _, tags = bio_tags("Book a table in New York tonight", [(16, 24, "city")])
        self.assertEqual(tags[4:6], ["B-city", "I-city"])

    def test_overlapping_spans_return_none(self):
        self.assertIsNone(bio_tags("Book a table in New York tonight", [(16, 24, "city"), (16, 19, "date")]))


class CleaningTest(unittest.TestCase):
    ROWS = [{"id": "1", "text": "Lakers tonight"}, {"id": "2", "text": "lakers TONIGHT"}, {"id": "3", "text": "Other"}]

    def test_dedupe_keeps_first_lowercased_text(self):
        kept, dropped = dedupe(self.ROWS)
        self.assertEqual([r["id"] for r in kept], ["1", "3"])
        self.assertEqual(dropped, 1)

    def test_drop_leaks_removes_banned_lowercased_texts(self):
        kept, dropped = drop_leaks(self.ROWS, {"other"})
        self.assertEqual([r["id"] for r in kept], ["1", "2"])
        self.assertEqual(dropped, 1)


class RepairBioTest(unittest.TestCase):
    def test_orphan_and_type_switch_inside_becomes_begin(self):
        tags = ["I-city", "I-city", "B-date", "I-city", "O", "I-date"]
        self.assertEqual(repair_bio(tags), ["B-city", "I-city", "B-date", "B-city", "O", "B-date"])


class SpansTest(unittest.TestCase):
    def test_spans_are_type_start_end_exclusive(self):
        tags = ["B-city", "I-city", "O", "B-date"]
        self.assertEqual(spans(tags), {("city", 0, 2), ("date", 3, 4)})

    def test_orphan_inside_tag_starts_a_span(self):
        self.assertEqual(spans(["I-date", "I-date"]), {("date", 0, 2)})


class SpanScoresTest(unittest.TestCase):
    def test_exact_match_scores_one(self):
        scores = span_scores([["B-city", "O"]], [["B-city", "O"]])
        self.assertEqual(scores["f1"], 1.0)
        self.assertEqual(scores["sentence_exact"], 1.0)
        self.assertEqual(scores["slots"]["city"]["support"], 1)

    def test_wrong_type_is_a_miss_and_a_false_alarm(self):
        scores = span_scores([["B-city", "O"]], [["B-date", "O"]])
        self.assertEqual(scores["f1"], 0.0)
        self.assertEqual(scores["slots"]["city"]["recall"], 0.0)
        self.assertEqual(scores["slots"]["date"]["precision"], 0.0)


class AddTyposTest(unittest.TestCase):
    TOKENS = ["hello", "world", "go", "Taylor", "swift", "!"]

    def test_same_seed_gives_same_noise(self):
        self.assertEqual(add_typos(self.TOKENS, random.Random(7), 1.0), add_typos(self.TOKENS, random.Random(7), 1.0))

    def test_zero_rate_changes_nothing(self):
        self.assertEqual(add_typos(self.TOKENS, random.Random(7), 0.0), self.TOKENS)

    def test_short_and_non_alphabetic_tokens_are_kept(self):
        noisy = add_typos(self.TOKENS, random.Random(3), 1.0)
        self.assertEqual(noisy[2], "go")
        self.assertEqual(noisy[5], "!")


class FeatureTest(unittest.TestCase):
    def test_features_use_word_shape_affixes_and_neighbours(self):
        feats = token_features(["Yankees", "game", "2026"], 0)
        self.assertEqual(feats["word"], "yankees")
        self.assertEqual(feats["shape"], "Xx")
        self.assertEqual(feats["suffix3"], "ees")
        self.assertTrue(feats["is_title"])
        self.assertEqual(feats["word+1"], "game")
        self.assertEqual(feats["word-1"], "<BOUNDARY>")

    def test_featurize_gives_one_dict_per_token(self):
        feats = featurize([{"tokens": ["a", "b"]}, {"tokens": ["c"]}])
        self.assertEqual([len(sentence) for sentence in feats], [2, 1])

    def test_predict_returns_one_tag_list_per_sentence(self):
        rows = [{"tokens": ["Paris", "tonight"], "tags": ["B-city", "O"]},
                {"tokens": ["Lakers", "game"], "tags": ["B-event_name", "I-event_name"]}]
        feats = featurize(rows)
        vec, X = fit_vectorizer(feats)
        model = LinearSVC(C=1).fit(X, [t for r in rows for t in r["tags"]])
        out = predict(vec, model, feats)
        self.assertEqual([len(sentence) for sentence in out], [2, 2])


class DictionaryTaggerTest(unittest.TestCase):
    def test_longest_remembered_span_wins(self):
        rows = [{"tokens": ["New", "York", "tonight"], "tags": ["B-city", "I-city", "O"]},
                {"tokens": ["New", "Year"], "tags": ["B-date", "I-date"]}]
        tagger = DictionaryTagger(rows)
        self.assertEqual(tagger.tag(["see", "new", "york", "tonight"]), ["O", "B-city", "I-city", "O"])
        self.assertEqual(tagger.tag(["new", "year"]), ["B-date", "I-date"])


class CleanSplitTest(unittest.TestCase):
    TRAIN = [{"text": "Book Paris", "tokens": [], "tags": []}]
    DEV = [{"text": "book PARIS", "tokens": [], "tags": []}, {"text": "New York", "tokens": [], "tags": []}]

    def test_first_split_only_dedupes(self):
        kept, duplicates, leakage = clean_split(self.TRAIN + self.TRAIN, [])
        self.assertEqual((len(kept), duplicates, leakage), (1, 1, 0))

    def test_later_split_drops_texts_seen_earlier(self):
        kept, duplicates, leakage = clean_split(self.DEV, [self.TRAIN])
        self.assertEqual([r["text"] for r in kept], ["New York"])
        self.assertEqual((duplicates, leakage), (0, 1))

    def test_slot_counts_are_sorted_by_slot(self):
        rows = [{"tags": ["B-date", "I-date", "B-city"]}, {"tags": ["B-city", "O"]}]
        self.assertEqual(slot_counts(rows), {"city": 2, "date": 1})


if __name__ == "__main__":
    unittest.main()
