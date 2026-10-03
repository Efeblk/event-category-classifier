import unittest

from classifier import make_models, normalize_text


class NormalizeTextTests(unittest.TestCase):
    def test_turkish_case_and_unicode_are_normalized(self):
        self.assertEqual(
            normalize_text("  IĞDIR\tİSTANBUL  ÇAĞRI ŞÖLENİ  "),
            "ığdır istanbul çağrı şöleni",
        )

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
        self.assertIsInstance(models["linear_svc"]["classifier"], LinearSVC)
        self.assertEqual(
            models["logistic_regression_unigram"]["tfidf"].ngram_range, (1, 1)
        )
        self.assertEqual(
            models["logistic_regression_bigram"]["tfidf"].ngram_range, (1, 2)
        )

    def test_every_candidate_fits_and_predicts_supported_labels(self):
        samples = {
            "concert": "canlı müzik konser sahne gitar",
            "theatre": "tiyatro sahne oyun oyuncu perde",
            "stand_up": "stand up komedi mizah kahkaha",
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
                self.assertEqual(len(predictions), 3)
                self.assertTrue(set(predictions).issubset(set(samples)))


if __name__ == "__main__":
    unittest.main()
