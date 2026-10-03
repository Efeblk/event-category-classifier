import unittest

from classifier import (
    ARTIFACT_TASK,
    DATASET_DOMAIN,
    LABELS,
    classify_request,
    make_models,
    normalize_text,
)


class NormalizeTextTests(unittest.TestCase):
    def test_turkish_case_and_unicode_are_normalized(self):
        self.assertEqual(
            normalize_text("  IĞDIR\tİSTANBUL  ÇAĞRI ŞÖLENİ  "),
            "ığdır istanbul çağrı şöleni",
        )

    def test_english_ascii_i_is_not_changed_to_dotless_i(self):
        self.assertEqual(normalize_text("I WANT LIVE MUSIC"), "i want live music")

    def test_none_becomes_empty_text(self):
        self.assertEqual(normalize_text(None), "")


class ModelDefinitionTests(unittest.TestCase):
    def test_required_candidates_and_ablation_are_present(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import LinearSVC

        models = make_models(seed=7)

        self.assertEqual(
            set(models),
            {
                "dummy_most_frequent",
                "multinomial_nb",
                "logistic_regression_unigram",
                "logistic_regression_bigram",
                "logistic_regression_char",
                "linear_svc",
            },
        )
        self.assertIsInstance(models["dummy_most_frequent"]["classifier"], DummyClassifier)
        self.assertIsInstance(models["multinomial_nb"]["classifier"], MultinomialNB)
        self.assertIsInstance(
            models["logistic_regression_unigram"]["classifier"], LogisticRegression
        )
        self.assertIsInstance(
            models["logistic_regression_bigram"]["classifier"], LogisticRegression
        )
        self.assertIsInstance(
            models["logistic_regression_char"]["classifier"], LogisticRegression
        )
        self.assertIsInstance(models["linear_svc"]["classifier"], LinearSVC)
        self.assertEqual(
            models["logistic_regression_unigram"]["tfidf"].ngram_range, (1, 1)
        )
        self.assertEqual(
            models["logistic_regression_bigram"]["tfidf"].ngram_range, (1, 2)
        )
        self.assertEqual(models["logistic_regression_char"]["tfidf"].analyzer, "char_wb")
        self.assertEqual(
            models["logistic_regression_char"]["tfidf"].ngram_range, (3, 5)
        )
        self.assertEqual(models["logistic_regression_char"]["tfidf"].min_df, 2)

    def test_every_candidate_fits_and_predicts_supported_labels(self):
        samples = {
            "concert": "canlı müzik konser sahne gitar",
            "theatre": "tiyatro sahne oyun oyuncu perde",
            "stand_up": "stand up komedi mizah kahkaha",
            "unclear": "kararsızım belki bir etkinlik olabilir",
        }
        texts = []
        labels = []
        for label, sample in samples.items():
            for index in range(6):
                texts.append(f"{sample} gösteri {index}")
                labels.append(label)

        for name, model in make_models(seed=11).items():
            with self.subTest(model=name):
                model.fit(texts, labels)
                predictions = model.predict(list(samples.values()))
                self.assertEqual(len(predictions), len(LABELS))
                self.assertTrue(set(predictions).issubset(set(samples)))


class ClassifyRequestTests(unittest.TestCase):
    def setUp(self):
        samples = {
            "concert": "live music concert guitar",
            "theatre": "theatre stage play actor",
            "stand_up": "stand up comedy jokes",
            "unclear": "maybe something undecided",
        }
        texts = []
        labels = []
        for label, sample in samples.items():
            for index in range(6):
                texts.append(f"{sample} example {index}")
                labels.append(label)
        self.model = make_models(seed=19)["logistic_regression_bigram"]
        self.model.fit(texts, labels)
        self.artifact = {
            "model": self.model,
            "model_name": "logistic_regression_bigram",
            "labels": sorted(LABELS),
            "dataset_sha256": "fixture",
            "seed": 19,
            "task": ARTIFACT_TASK,
            "dataset_domain": DATASET_DOMAIN,
            "provenance": "generated_bootstrap",
        }

    def test_known_request_uses_the_trained_model_and_reports_probabilities(self):
        result = classify_request(self.artifact, "I want a live music concert")

        self.assertEqual(result["label"], "concert")
        self.assertEqual(set(result["model_probabilities"]), set(LABELS))

    def test_zero_vocabulary_request_returns_unclear(self):
        result = classify_request(self.artifact, "qxzv blorpt nymwax")

        self.assertEqual(
            result, {"label": "unclear", "reason": "unknown_terms"}
        )

    def test_old_artifact_task_is_rejected(self):
        self.artifact["task"] = "event_description_classification"

        with self.assertRaisesRegex(ValueError, "task"):
            classify_request(self.artifact, "live music concert")

    def test_blank_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-whitespace"):
            classify_request(self.artifact, "  \t ")


if __name__ == "__main__":
    unittest.main()
