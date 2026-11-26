"""Temporary-fixture tests for the one-time old-test evaluation."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib

from classifier import LABELS
import final_test
import prepare_data
from train import train


TEST_ARTICLE = "9001"
TRAIN_ARTICLE = "8001"
PREFIX = "x" * 1200
SUFFIX = "y" * 1200


def development_rows():
    """Two train and one dev example per technique, article-disjoint."""
    rows = []
    for partition in ("train", "dev"):
        for index, label in enumerate(LABELS):
            for repeat in range(2 if partition == "train" else 1):
                text = label.replace("_", " ") + " example"
                start = repeat * 200
                rows.append({
                    "id": f"{partition}-{index}-{repeat}", "article_id": f"{partition}-article-{index}",
                    "text": text, "context": f"{partition} context for {text}", "label": label,
                    "start": start, "end": start + len(text), "partition": partition,
                    "source_partition": "official_train",
                })
    return rows


def make_data_dir(root: Path, test_rows: int = 2) -> Path:
    """Create raw test/train articles, a split manifest and development rows."""
    data = root / "data"
    (data / "raw" / "articles").mkdir(parents=True)
    (data / "raw" / "annotations").mkdir(parents=True)
    (data / "cleaned").mkdir(parents=True)
    article = PREFIX + "Not false!" + SUFFIX
    (data / "raw" / "articles" / f"article{TEST_ARTICLE}.txt").write_bytes(article.encode("utf-8"))
    (data / "raw" / "annotations" / f"article{TEST_ARTICLE}.task2-TC.labels").write_text(
        f"{TEST_ARTICLE}\t{LABELS[0]}\t1200\t1210\n"
        f"{TEST_ARTICLE}\t{LABELS[0]}\t1200\t1210\n"  # exact duplicate, removed
        f"{TEST_ARTICLE}\t{LABELS[1]}\t1200\t1210\n",  # distinct technique, kept
        encoding="utf-8", newline="\n")
    (data / "raw" / "articles" / f"article{TRAIN_ARTICLE}.txt").write_bytes(b"Train only text here.")
    (data / "raw" / "annotations" / f"article{TRAIN_ARTICLE}.task2-TC.labels").write_text(
        f"{TRAIN_ARTICLE}\t{LABELS[2]}\t0\t5\n", encoding="utf-8", newline="\n")
    manifest = {
        "source_partition": "official_train", "seed": 42,
        "article_partitions": {TRAIN_ARTICLE: "train", TEST_ARTICLE: "test"},
        "article_counts": {"train": 1, "dev": 0, "test": 1},
        "row_counts": {"train": 1, "dev": 0, "test": test_rows},
    }
    (data / "cleaned" / "splits.json").write_text(json.dumps(manifest), encoding="utf-8")
    (data / "cleaned" / "development.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in development_rows()),
        encoding="utf-8", newline="\n")
    return data


def train_artifacts(root: Path, data: Path) -> tuple[Path, Path, Path]:
    reports = root / "reports"
    model = root / "artifacts" / "model.joblib"
    train(data / "cleaned" / "development.jsonl", model, reports, allow_small_prototype=True)
    return root / "artifacts" / "method_models.joblib", model, reports / "metrics.json"


def assert_scores_present(test, model_scores):
    for key in ("accuracy", "micro_f1", "macro_f1"):
        test.assertIsInstance(model_scores[key], float)
    test.assertIsNotNone(model_scores["per_class"])


class BuildTestRowsTests(unittest.TestCase):
    def test_uses_only_test_articles_with_thousand_character_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = make_data_dir(Path(temporary))
            rows = final_test.build_test_rows(data)
        self.assertEqual({row["article_id"] for row in rows}, {TEST_ARTICLE})
        self.assertEqual({row["partition"] for row in rows}, {"test"})
        self.assertEqual(len(rows), 2)
        self.assertEqual(sorted(row["label"] for row in rows), sorted([LABELS[0], LABELS[1]]))
        for row in rows:
            self.assertEqual(row["text"], "Not false!")
            # Exactly 1,000 original characters on each side; no spaces in the filler.
            self.assertEqual(len(row["context"]), 1000 + len("Not false!") + 1000)
            self.assertTrue(row["context"].startswith("x" * 1000 + "Not false!"))

    def test_row_count_mismatch_raises_readable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = make_data_dir(Path(temporary), test_rows=3)
            with self.assertRaisesRegex(ValueError, "differ from the original split manifest"):
                final_test.build_test_rows(data)

    def test_does_not_write_under_cleaned_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = make_data_dir(Path(temporary))
            before = sorted(path.name for path in (data / "cleaned").iterdir())
            final_test.build_test_rows(data)
            self.assertEqual(sorted(path.name for path in (data / "cleaned").iterdir()), before)


class FinalTestTests(unittest.TestCase):
    def test_refuses_non_development_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = make_data_dir(root)
            models, model, metrics = train_artifacts(root, data)
            artifacts = joblib.load(models)
            for artifact in artifacts.values():
                artifact["provenance"] = dict(artifact["provenance"],
                                              evaluation_scope="historical_baseline", context_margin=350)
            historical = root / "historical.joblib"
            joblib.dump(artifacts, historical)
            with self.assertRaisesRegex(ValueError, "development-only"):
                final_test.run_evaluation(historical, model, data, root / "out.json", metrics)

    def test_report_is_null_free_caveated_and_not_overwritten_without_force(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = make_data_dir(root)
            models, model, metrics = train_artifacts(root, data)
            report = root / "out" / "test_evaluation.json"
            payload = final_test.run_evaluation(models, model, data, report, metrics)

            saved = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(saved, json.loads(json.dumps(payload, ensure_ascii=False)))
            self.assertEqual(saved["schema_version"], 1)
            self.assertEqual(saved["evaluation_scope"], "old_test_optimistic_one_time")
            self.assertEqual(saved["partition"], "test")
            self.assertEqual((saved["rows"], saved["articles"], saved["context_margin"]), (2, 1, 1000))
            self.assertIn("Optimistically biased", saved["caveat"])
            self.assertIn("47.32%", saved["historical_baseline_test"]["note"])
            self.assertEqual(set(saved["models"]), {"selected_overall", "naive_bayes", "logistic_regression", "linear_svm"})
            for model_scores in saved["models"].values():
                assert_scores_present(self, model_scores)
            assert_scores_present(self, saved["majority_baseline"])
            self.assertEqual(saved["dev_test_overlap"]["test_articles_in_train_or_dev"], 0)
            self.assertTrue((report.parent / "test_confusion_matrix.png").is_file())
            self.assertIsNotNone(saved["development_metrics_check"])
            self.assertEqual(report.read_bytes().count(b"\r\n"), 0)

            with self.assertRaisesRegex(ValueError, "one-time"):
                final_test.run_evaluation(models, model, data, report, metrics)
            replaced = final_test.run_evaluation(models, model, data, report, metrics, force=True)
            self.assertEqual(replaced["rows"], 2)


if __name__ == "__main__":
    unittest.main()
