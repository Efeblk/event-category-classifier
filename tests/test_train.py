import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import train as training


def make_rows(count=1_200, grouped=False):
    labels = training.LABELS
    rows = []
    for index in range(count):
        label = labels[index % len(labels)]
        group_number = index // 2 if grouped else index
        rows.append(
            {
                "id": f"row-{index:04d}",
                "text": f"unique event text {index:04d} for {label}",
                "label": label,
                "group_id": f"group-{group_number:04d}",
                "title": f"Event {index}",
                "source": "fixture",
                "source_url": f"https://example.test/{index}",
            }
        )
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=training.REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class ReadCsvTests(unittest.TestCase):
    def test_read_csv_accepts_a_valid_minimum_dataset(self):
        rows = make_rows(count=1_002)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            write_csv(path, rows)

            loaded = training.read_csv(path)

        self.assertEqual(loaded, rows)

    def test_read_csv_allows_an_explicit_small_prototype(self):
        rows = make_rows(count=40)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            write_csv(path, rows)

            loaded = training.read_csv(path, min_rows=40)

        self.assertEqual(loaded, rows)

    def test_read_csv_rejects_repeated_normalized_text(self):
        rows = make_rows(count=1_002)
        rows[1]["text"] = f"  {rows[0]['text']}  "
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            write_csv(path, rows)

            with self.assertRaisesRegex(ValueError, "not deduplicated"):
                training.read_csv(path)

    def test_read_csv_rejects_mixed_labels_in_one_group(self):
        rows = make_rows(count=1_002)
        rows[1]["group_id"] = rows[0]["group_id"]
        self.assertNotEqual(rows[0]["label"], rows[1]["label"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            write_csv(path, rows)

            with self.assertRaisesRegex(ValueError, "group.*label|label.*group"):
                training.read_csv(path)


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_group_disjoint_and_near_60_20_20(self):
        rows = make_rows(grouped=True)

        first = training.split_data(rows, seed=29)
        second = training.split_data(rows, seed=29)

        self.assertEqual(first, second)
        all_indices = [index for indices in first.values() for index in indices]
        self.assertEqual(len(all_indices), len(rows))
        self.assertEqual(len(set(all_indices)), len(rows))
        group_sets = {
            name: {rows[index]["group_id"] for index in indices}
            for name, indices in first.items()
        }
        self.assertFalse(group_sets["train"] & group_sets["validation"])
        self.assertFalse(group_sets["train"] & group_sets["test"])
        self.assertFalse(group_sets["validation"] & group_sets["test"])
        for name, expected_share in (("train", 0.6), ("validation", 0.2), ("test", 0.2)):
            self.assertAlmostEqual(len(first[name]) / len(rows), expected_share, delta=0.03)
            self.assertEqual(
                {rows[index]["label"] for index in first[name]}, set(training.LABELS)
            )


class FakeModel:
    def __init__(self, predict_correctly):
        self.predict_correctly = predict_correctly
        self.fit_calls = []

    def fit(self, texts, labels):
        self.fit_calls.append((list(texts), list(labels)))
        return self

    def predict(self, texts):
        if not self.predict_correctly:
            return ["concert"] * len(texts)
        return [text.split(":", 1)[0] for text in texts]


class TrainingFlowTests(unittest.TestCase):
    def test_selection_uses_validation_and_saved_model_is_not_refit(self):
        rows = []
        for part in ("train", "validation", "test"):
            for label in training.LABELS:
                index = len(rows)
                rows.append(
                    {
                        "id": str(index),
                        "text": f"{label}:{part}",
                        "label": label,
                        "group_id": f"group-{index}",
                        "title": "",
                        "source": "",
                        "source_url": "",
                    }
                )
        class_count = len(training.LABELS)
        parts = {
            "train": list(range(0, class_count)),
            "validation": list(range(class_count, 2 * class_count)),
            "test": list(range(2 * class_count, 3 * class_count)),
        }
        good = FakeModel(predict_correctly=True)
        bad = FakeModel(predict_correctly=False)
        models = {"good": good, "bad": bad}

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            source.write_text("fixture", encoding="utf-8")
            destination = Path(directory) / "model.joblib"
            with (
                mock.patch.object(training, "read_csv", return_value=rows) as read_csv,
                mock.patch.object(training, "split_data", return_value=parts),
                mock.patch.object(training, "make_models", return_value=models),
                mock.patch.object(training, "cross_validate", return_value={}),
                mock.patch.object(training, "_write_reports") as write_reports,
                mock.patch("joblib.dump") as dump,
            ):
                metadata = training.train(source, destination, Path(directory) / "reports")

        self.assertEqual(metadata["selection"]["selected_model"], "good")
        self.assertEqual(metadata["selection"]["fit_split"], "train")
        self.assertFalse(metadata["selection"]["final_refit"])
        self.assertEqual(len(good.fit_calls), 1)
        self.assertEqual(len(bad.fit_calls), 1)
        self.assertEqual(
            good.fit_calls[0][0], [row["text"] for row in rows[:class_count]]
        )
        read_csv.assert_called_once_with(source, min_rows=training.MIN_ROWS)
        artifact = dump.call_args.args[0]
        self.assertIs(artifact["model"], good)
        self.assertEqual(artifact["model_name"], "good")
        self.assertEqual(artifact["task"], "user_request_classification")
        self.assertEqual(artifact["dataset_domain"], "user_requests")
        self.assertEqual(artifact["provenance"], "generated_bootstrap")
        self.assertFalse(artifact["course_dataset_ready"])
        self.assertEqual(metadata["evaluation_scope"], "raw_model")
        self.assertFalse(metadata["course_dataset_ready"])
        write_reports.assert_called_once()


class CrossValidationTests(unittest.TestCase):
    def test_every_model_gets_grouped_fold_scores(self):
        rows = make_rows(count=200, grouped=True)
        results = training.cross_validate(rows, list(range(len(rows))), seed=3)

        self.assertEqual(set(results), set(training.make_models()))
        for name, result in results.items():
            with self.subTest(model=name):
                self.assertEqual(len(result["fold_macro_f1"]), 5)
                self.assertTrue(0 <= result["macro_f1_mean"] <= 1)
                self.assertTrue(0 <= result["macro_f1_std"] <= 1)


if __name__ == "__main__":
    unittest.main()
