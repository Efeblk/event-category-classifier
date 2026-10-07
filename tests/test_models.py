import unittest
from classifier import LABELS, OUTPUT_LABELS, ARTIFACT_TASK, DATASET_DOMAIN, make_models, normalize_text, classify_request, _validated_model


def fixture_artifact():
    model = make_models()["logistic_regression_word"]
    model.fit([label.replace("_", " ")+" fixture" for label in LABELS]*2, list(LABELS)*2)
    return {"model": model, "model_name": "logistic_regression_word", "labels": LABELS,
            "dataset_sha256": "fixture", "seed": 42, "task": ARTIFACT_TASK,
            "dataset_domain": DATASET_DOMAIN, "provenance": "fixture"}

class ModelTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_text(" IĞDIR İSTANBUL I "), "ığdır istanbul ı")
        self.assertEqual(normalize_text(None), "")
    def test_candidates(self):
        models = make_models()
        self.assertEqual(len(models),7)
        for name, model in models.items():
            model.fit([x.replace("_"," ")+" fixture" for x in LABELS]*2,list(LABELS)*2)
            self.assertEqual(set(model.classes_),set(LABELS))
            self.assertIn(model.predict(["alarm set fixture"])[0],LABELS)
    def test_schema_and_abstention(self):
        artifact = fixture_artifact()
        self.assertEqual(len(LABELS),60)
        self.assertNotIn("unclear",LABELS)
        self.assertEqual(len(OUTPUT_LABELS),61)
        self.assertEqual(classify_request(artifact,"qxzv") , {"label":"unclear","reason":"unknown_terms"})
        self.assertIn(classify_request(artifact,"alarm set fixture")["label"],LABELS)
        for key, value in (("task","old"),("dataset_domain","old"),("labels",LABELS[:-1])):
            with self.assertRaises(ValueError): _validated_model(artifact | {key:value})
        with self.assertRaises(ValueError): _validated_model({})
        with self.assertRaises(ValueError): classify_request(artifact," ")
