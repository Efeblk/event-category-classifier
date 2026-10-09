import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from classifier import (
    ARTIFACT_TASK, ARTIFACT_VERSION, DATASET_DOMAIN, LABELS, LABEL_DESCRIPTIONS,
    OUTPUT_LABELS, _validated_model, classify_request, make_models, model_input,
    _feature_names, normalize_text, structure_features,
)


def fixture_inputs():
    return [model_input(label.replace("_", " ") + " example", "Context for " + label.replace("_", " "))
            for label in LABELS] * 2


def fixture_artifact(name="logistic_regression_word"):
    model = make_models()[name]
    model.fit(fixture_inputs(), list(LABELS) * 2)
    return {
        "schema_version": ARTIFACT_VERSION, "model": model, "model_name": name,
        "labels": LABELS, "dataset_sha256": "a" * 64, "code_sha256": "b" * 64,
        "seed": 42, "task": ARTIFACT_TASK, "dataset_domain": DATASET_DOMAIN,
        "provenance": {"fit_partition": "train", "source_partition": "official_train",
                       "evaluation_scope": "dev_only", "context_margin": 1000},
    }


class ModelTests(unittest.TestCase):
    def test_english_normalization_retains_punctuation(self):
        self.assertEqual(normalize_text(" ＦＥＡＲ\n I! "), "fear i!")
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text(" DON'T\ttrust them. "), "don't trust them.")

    def test_predeclared_candidates_fit_all_classes(self):
        models = make_models()
        self.assertEqual(len(models), 10)
        baseline = make_models(profile="baseline")
        self.assertEqual(len(baseline), 9)
        self.assertNotIn("logistic_regression_hybrid_structure", baseline)
        for name, model in models.items():
            with self.subTest(model=name):
                model.fit(fixture_inputs(), list(LABELS) * 2)
                self.assertEqual(set(model.classes_), set(LABELS))
                self.assertIn(model.predict([model_input("loaded language example", "Context for loaded language")])[0], LABELS)

    def test_fixed_labels_and_descriptions(self):
        self.assertEqual(len(LABELS), 14)
        self.assertEqual(set(LABEL_DESCRIPTIONS), set(LABELS))
        self.assertIn("Bandwagon,Reductio_ad_hitlerum", LABELS)
        self.assertNotIn("unclear", LABELS)
        self.assertEqual(len(OUTPUT_LABELS), 15)

    def test_schema_and_unknown_terms_abstention(self):
        artifact = fixture_artifact()
        self.assertEqual(classify_request(artifact, "qxzv"), {"label": "unclear", "reason": "unknown_terms"})
        prediction = classify_request(artifact, "loaded language example")
        self.assertIn(prediction["label"], LABELS)
        self.assertAlmostEqual(sum(prediction["model_probabilities"].values()), 1, places=4)
        self.assertTrue(prediction["top_features"])
        self.assertTrue(all(feature["weight"] > 0 for feature in prediction["top_features"]))
        values = (
            ("task", "old"), ("dataset_domain", "old"), ("labels", LABELS[:-1]),
            ("schema_version", 99), ("dataset_sha256", "fixture"), ("code_sha256", None),
            ("seed", True), ("provenance", {"fit_partition": "train+dev"}),
            ("provenance", {"fit_partition": "train", "evaluation_scope": "dev_only", "context_margin": 350}),
            ("provenance", {"fit_partition": "train", "evaluation_scope": "dev_only"}),
            ("provenance", {"fit_partition": "train", "evaluation_scope": [], "context_margin": 1000}),
            ("provenance", {"fit_partition": "train", "evaluation_scope": {}, "context_margin": 1000}),
        )
        for key, value in values:
            with self.subTest(key=key), self.assertRaises(ValueError):
                _validated_model(artifact | {key: value})
        with self.assertRaises(ValueError):
            _validated_model({})
        with self.assertRaises(ValueError):
            classify_request(artifact, " ")

    def test_context_is_an_independent_feature_channel(self):
        artifact = fixture_artifact("logistic_regression_word_context")
        self.assertEqual(classify_request(artifact, "qxzv"), {"label": "unclear", "reason": "unknown_terms"})
        self.assertEqual(classify_request(artifact, "qxzv", "Context for loaded language"),
                         {"label": "unclear", "reason": "unknown_terms"})
        prediction = classify_request(artifact, "example", "Context for loaded language")
        self.assertIn(prediction["label"], LABELS)
        self.assertTrue(any(feature["feature"].startswith("context:") for feature in prediction["top_features"]))

    def test_svm_does_not_invent_probabilities(self):
        prediction = classify_request(fixture_artifact("linear_svm_word"), "loaded language example")
        self.assertNotIn("model_probabilities", prediction)

    def test_artifact_round_trip_preserves_top_level_transformers(self):
        import joblib

        artifact = fixture_artifact("logistic_regression_hybrid_structure")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(artifact, path)
            restored = joblib.load(path)
        self.assertEqual(classify_request(artifact, "loaded language example", "Context"),
                         classify_request(restored, "loaded language example", "Context"))

    def test_structural_values_have_no_labels_or_article_identifiers(self):
        records = [model_input("NO! NO!", "NO! NO! Then NO! NO!")]
        first = structure_features(records)[0]
        self.assertEqual(len(first), 22)
        self.assertEqual(first["uppercase_fraction"], 1)
        self.assertEqual(first["exclamation_mark"], 1)
        self.assertEqual(first["repeated_in_context"], 1)
        self.assertEqual(first, structure_features([records[0] | {"label": LABELS[0], "article_id": "private"}])[0])

    def test_numeric_structure_alone_does_not_prevent_abstention(self):
        artifact = fixture_artifact("logistic_regression_hybrid_structure")
        self.assertGreater(artifact["model"].named_steps["features"].transform([model_input("\U0001f995")]).nnz, 0)
        self.assertEqual(classify_request(artifact, "\U0001f995"), {"label": "unclear", "reason": "unknown_terms"})
        names = _feature_names(artifact["model"].named_steps["features"])
        self.assertIn("structure: uppercase_fraction", names)
        self.assertEqual(classify_request(artifact, "\U0001f995", "loaded language example"),
                         {"label": "unclear", "reason": "unknown_terms"})

    def test_context_cannot_supply_missing_excerpt_vocabulary(self):
        for name in ("logistic_regression_word_context", "linear_svm_word_context",
                     "logistic_regression_hybrid_structure"):
            with self.subTest(model=name):
                artifact = fixture_artifact(name)
                self.assertGreater(artifact["model"].named_steps["features"].transform(
                    [model_input("qxzv", "loaded language example")]).nnz, 0)
                self.assertEqual(classify_request(artifact, "qxzv", "loaded language example"),
                                 {"label": "unclear", "reason": "unknown_terms"})
                self.assertIn(classify_request(artifact, "loaded language example", "loaded language example")["label"], LABELS)

    def test_prediction_requires_string_excerpt_and_context(self):
        artifact = fixture_artifact()
        for text, context in ((123, ""), (None, ""), ("example", []), ("example", None)):
            with self.subTest(text=text, context=context), self.assertRaises(ValueError):
                classify_request(artifact, text, context)

    def test_direct_prediction_validates_unicode_without_changing_valid_text(self):
        artifact = fixture_artifact("logistic_regression_hybrid_structure")
        invalid = ("loaded\x00 language", "loaded language\ud800", "\udfff", "\u200b\u200d", "\u034f\ufe0f")
        for text in invalid:
            with self.subTest(excerpt=ascii(text)), self.assertRaises(ValueError):
                classify_request(artifact, text, "Context for loaded language")
        for context in invalid:
            with self.subTest(context=ascii(context)), self.assertRaises(ValueError):
                classify_request(artifact, "loaded language example", context)
        text = "loaded language example cafe\u0301 \U0001f468\u200d\U0001f469 \u2600\ufe0f\n\ttext!"
        context = "Context for loaded language. More\r\ncontext \u200bhere."
        original = model_input(text, context)
        features = artifact["model"].named_steps["features"]
        with patch.object(features, "transform", wraps=features.transform) as transformed:
            self.assertIn(classify_request(artifact, text, context)["label"], LABELS)
        self.assertTrue(transformed.called)
        self.assertTrue(all(call.args[0] == [original] for call in transformed.call_args_list))


if __name__ == "__main__":
    unittest.main()
