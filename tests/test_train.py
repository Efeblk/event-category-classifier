import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from classifier import LABELS
from train import comparison_records, read_data, select_model, split_data, train


def fixture_rows():
    """One row per intent in each official partition: 180 rows."""
    rows = []
    for i, part in enumerate(("train", "dev", "test")):
        for j, label in enumerate(LABELS):
            rows.append({
                "id": str(i * 60 + j),
                "text": label.replace("_", " ") + " fixture",
                "intent": label,
                "scenario": label.split("_")[0],
                "partition": part,
            })
    return rows


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class TrainTests(unittest.TestCase):
    def test_official_split_and_tiny_class(self):
        rows = fixture_rows()
        parts = split_data(rows)
        self.assertEqual(parts, split_data(rows, 99))
        self.assertEqual([len(indices) for indices in parts.values()], [60, 60, 60])
        self.assertEqual(set(sum(parts.values(), [])), set(range(180)))
        # Test may miss an intent (as cooking_query does); train may not.
        self.assertEqual(len(split_data(rows[:-1])["test"]), 59)
        with self.assertRaises(ValueError):
            split_data(rows[1:])
        invalid = [dict(row) for row in rows]
        invalid[0]["partition"] = "other"
        with self.assertRaises(ValueError):
            split_data(invalid)

    def test_data_validation_preserves_official_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            rows = fixture_rows()
            write_rows(path, rows)
            self.assertEqual(len(read_data(path, min_rows=1)), 180)
            with self.assertRaisesRegex(ValueError, "1000"):
                read_data(path)
            bad_values = (
                ("id", rows[0]["id"]),
                ("intent", "unclear"),
                ("scenario", "wrong"),
                ("partition", "validation"),
                ("text", " "),
            )
            for key, value in bad_values:
                with self.subTest(key=key):
                    bad = [dict(row) for row in rows]
                    bad[1][key] = value
                    write_rows(path, bad)
                    with self.assertRaises(ValueError):
                        read_data(path, min_rows=1)

    def test_selection_ignores_test(self):
        evaluations = {
            "a": {"dev": {"macro_f1": .8}, "test": {"macro_f1": 0}},
            "b": {"dev": {"macro_f1": .2}, "test": {"macro_f1": 1}},
        }
        self.assertEqual(select_model(evaluations), "a")

    def test_comparison_is_test_only_deterministic(self):
        rows = fixture_rows()
        indices = split_data(rows)["test"]
        records = comparison_records(rows, indices)
        self.assertEqual(records, comparison_records(rows, indices))
        self.assertEqual(len(records), 40)
        self.assertEqual(len({record["id"] for record in records}), 40)
        self.assertTrue({record["id"] for record in records} <= {rows[i]["id"] for i in indices})

    def test_flow_fits_train_and_predicts_test_after_selection(self):
        events = []

        class Model:
            def fit(self, x, y):
                events.append(("fit", len(x)))
                return self

            def predict(self, x):
                events.append(("predict", len(x)))
                return list(LABELS)

            def set_params(self, **kwargs):
                return self

        rows = fixture_rows()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("train.read_data", return_value=rows), \
                patch("train.make_models", return_value={"logistic_regression_word": Model()}), \
                patch("sklearn.base.clone", side_effect=lambda model: model), \
                patch("joblib.dump"), \
                patch("train.plot_confusion"):
            path = Path(tmp) / "data"
            path.write_text("fixture")
            result = train(path, Path(tmp) / "model.joblib", tmp, allow_small_prototype=True)

        # The balanced and unbalanced candidates fit on train and predict dev,
        # then both predict test. No fit happens after selection.
        self.assertEqual(events, [("fit", 60), ("predict", 60)] * 2 + [("predict", 60)] * 2)
        self.assertEqual(result["selection"]["fit_split"], "train")
        self.assertFalse(result["selection"]["test_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
