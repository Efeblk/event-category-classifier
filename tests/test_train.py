import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from classifier import LABELS, validate_unicode_text
from train import (
    _code_hashes, comparison_records, dev_records, duplicate_audit, read_data, scores,
    select_model, split_data, train,
)


def fixture_rows(profile="baseline"):
    """Two train examples per class and one dev/test, all article disjoint."""
    rows = []
    for part in (("train", "dev", "test") if profile == "baseline" else ("train", "dev")):
        for j, label in enumerate(LABELS):
            for repeat in range(2 if part == "train" else 1):
                text = label.replace("_", " ") + " example"
                rows.append({
                    "id": f"{part}-{j}-{repeat}", "article_id": f"{part}-article-{j}",
                    "text": text, "context": f"{part} context for {text}",
                    "label": label, "start": repeat * 200, "end": repeat * 200 + len(text),
                    "partition": part, "source_partition": "official_train",
                })
    return rows


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TrainTests(unittest.TestCase):
    def test_code_provenance_includes_preparation_and_json_validation(self):
        digest, script_hashes = _code_hashes()
        self.assertEqual(len(digest), 64)
        self.assertEqual(set(script_hashes), {"classifier.py", "train.py", "prepare_data.py", "json_validation.py"})

    def test_grouped_split_allows_absent_dev_test_classes(self):
        rows = fixture_rows()
        parts = split_data(rows, profile="baseline")
        self.assertEqual(parts, split_data(rows, 99, profile="baseline"))
        self.assertEqual([len(indices) for indices in parts.values()], [28, 14, 14])
        self.assertEqual(set(sum(parts.values(), [])), set(range(56)))
        self.assertEqual(len(split_data(rows[:-1], profile="baseline")["test"]), 13)
        with self.assertRaisesRegex(ValueError, "all 14"):
            split_data(rows[2:], profile="baseline")

    def test_article_and_identical_input_leakage_rejected(self):
        rows = fixture_rows("development")
        invalid = [dict(row) for row in rows]
        invalid[-1]["article_id"] = rows[0]["article_id"]
        with self.assertRaisesRegex(ValueError, "article crosses"):
            split_data(invalid)
        invalid = [dict(row) for row in rows]
        invalid[-1]["text"] = "  " + rows[0]["text"].upper()
        invalid[-1]["context"] = rows[0]["context"].upper()
        with self.assertRaisesRegex(ValueError, "excerpt/context"):
            split_data(invalid)

    def test_data_validation_retains_distinct_multiple_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            rows = fixture_rows()
            rows.append(rows[0] | {"id": "multi-label", "label": LABELS[1]})
            write_rows(path, rows)
            self.assertEqual(len(read_data(path, min_rows=1, profile="baseline")), 57)
            self.assertEqual(duplicate_audit(rows)["multi_label_exact_spans"], 1)
            with self.assertRaisesRegex(ValueError, "1000"):
                read_data(path)
            bad_values = (
                ("id", rows[0]["id"]), ("label", "unclear"), ("article_id", " "),
                ("source_partition", "official_test"), ("partition", "validation"),
                ("text", " "), ("context", " "), ("start", -1), ("end", True),
            )
            for key, value in bad_values:
                with self.subTest(key=key):
                    bad = [dict(row) for row in rows]
                    bad[1][key] = value
                    write_rows(path, bad)
                    with self.assertRaises(ValueError):
                        read_data(path, min_rows=1, profile="baseline")
            write_rows(path, rows + [rows[0] | {"id": "exact-duplicate"}])
            with self.assertRaisesRegex(ValueError, "Exact duplicate"):
                read_data(path, min_rows=1, profile="baseline")
            path.write_text("invalid json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                read_data(path, min_rows=1)

    def test_selection_ignores_test_and_breaks_ties_deterministically(self):
        evaluations = {
            "b": {"dev": {"macro_f1": .8}, "test": {"macro_f1": 0}},
            "a": {"dev": {"macro_f1": .8}, "test": {"macro_f1": .2}},
            "c": {"dev": {"macro_f1": .2}, "test": {"macro_f1": 1}},
        }
        self.assertEqual(select_model(evaluations), "a")
        self.assertEqual(select_model(evaluations, ["b"]), "b")

    def test_training_rejects_invalid_unicode_before_creating_models(self):
        invalid_values = ("\ud800", "words\udfff", "words\x00", "words\x1b", "\u200b", "\ufe0f")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "development.jsonl"
            for field in ("text", "context"):
                for value in invalid_values:
                    with self.subTest(field=field, value=ascii(value)):
                        rows = fixture_rows("development")
                        rows[0][field] = value
                        if field == "text":
                            rows[0]["end"] = rows[0]["start"] + len(value)
                        write_rows(path, rows)
                        location = "data line 1" if any(0xD800 <= ord(character) <= 0xDFFF for character in value) else "Row 1 " + field
                        with patch("train.make_models") as candidates, self.assertRaisesRegex(ValueError, location):
                            train(path, Path(directory) / "model.joblib", directory, allow_small_prototype=True)
                        candidates.assert_not_called()

    def test_training_rejects_ambiguous_or_nonfinite_json_before_fitting(self):
        rows = fixture_rows("development")
        source = "".join(json.dumps(row) + "\n" for row in rows)
        first_row = json.dumps(rows[0])
        malformed = (
            first_row[:-1] + ', "label": "Doubt"}',
            first_row[:-1] + ', "weight": NaN}',
            first_row[:-1] + ', "weight": Infinity}',
            first_row[:-1] + ', "weight": 1e999}',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "development.jsonl"
            for first in malformed:
                with self.subTest(first=first[-40:]):
                    path.write_text(first + "\n" + source[len(first_row) + 1:], encoding="utf-8")
                    with patch("train.make_models") as candidates, self.assertRaisesRegex(ValueError, "data line 1"):
                        train(path, Path(directory) / "model.joblib", directory, allow_small_prototype=True)
                    candidates.assert_not_called()

    def test_training_requires_utf8_jsonl_and_keeps_unicode_line_separators(self):
        rows = fixture_rows("development")
        rows[0]["context"] += "\u2028more\u2029context"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "development.jsonl"
            content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            for encoding in ("utf-16", "utf-32"):
                with self.subTest(encoding=encoding):
                    path.write_bytes(content.encode(encoding))
                    with self.assertRaisesRegex(ValueError, "data line 1"):
                        read_data(path, min_rows=1)
            path.write_bytes(content.encode("utf-8"))
            self.assertEqual(read_data(path, min_rows=1)[0]["context"], rows[0]["context"])

    def test_training_preserves_visible_unicode_and_line_breaks(self):
        text = "Human cafe\u0301 \U0001f468\u200d\U0001f469 \u2600\ufe0f\r\n\twords!"
        context = "Nearby\r\ncontext with \u200bvisible words."
        rows = fixture_rows("development")
        rows[0].update(text=text, context=context, end=rows[0]["start"] + len(text))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "development.jsonl"
            write_rows(path, rows)
            prepared = read_data(path, min_rows=1)
        self.assertEqual(prepared[0]["text"], text)
        self.assertEqual(prepared[0]["context"], context)

    def test_training_preserves_source_c1_controls_but_inference_remains_strict(self):
        rows = fixture_rows("development")
        rows[0]["context"] = "Original source \u009dquoted words\u009d remain unchanged."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "development.jsonl"
            write_rows(path, rows)
            prepared = read_data(path, min_rows=1)
        self.assertEqual(prepared[0]["context"], rows[0]["context"])
        with self.assertRaisesRegex(ValueError, "unsupported control"):
            validate_unicode_text(rows[0]["context"], "Context")
        for value in ("source\x00", "source\x1b", "\u009d"):
            with self.subTest(value=ascii(value)), self.assertRaises(ValueError):
                validate_unicode_text(value, "Source", allow_c1_controls=True)

    def test_fixed_macro_f1_counts_absent_classes_as_zero(self):
        result = scores([LABELS[0]], [LABELS[0]])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["micro_f1"], 1)
        self.assertAlmostEqual(result["macro_f1"], 1 / 14)
        self.assertEqual(len(result["confusion_matrix"]), 14)

    def test_default_demo_and_uniform_comparison_are_dev_only(self):
        rows = fixture_rows()
        parts = split_data(rows, profile="baseline")
        records = comparison_records(rows, parts["dev"])
        self.assertEqual(records, comparison_records(rows, parts["dev"]))
        self.assertEqual(len(records), 14)
        self.assertTrue({record["id"] for record in records} <= {rows[i]["id"] for i in parts["dev"]})
        examples = dev_records(rows, parts["dev"])
        self.assertEqual(len(examples), 14)
        self.assertTrue({record["id"] for record in examples} <= {rows[i]["id"] for i in parts["dev"]})
        with self.assertRaises(ValueError):
            comparison_records(rows, parts["train"])
        with self.assertRaises(ValueError):
            comparison_records(rows, parts["test"])
        with self.assertRaises(ValueError):
            dev_records(rows, parts["test"])

    def test_fit_train_and_freeze_all_choices_before_test(self):
        events = []

        class Model:
            def fit(self, x, y):
                self.classes_ = list(LABELS)
                events.append(("fit", {record["context"].split()[0] for record in x}, len(x)))
                return self

            def predict(self, x):
                events.append(("predict", {record["context"].split()[0] for record in x}, len(x)))
                return list(LABELS)

        def record_selection(evaluations, names=None):
            events.append(("select",))
            return select_model(evaluations, names)

        models = {name: Model() for name in ("multinomial_nb_word", "logistic_regression_word", "linear_svm_word")}
        rows = fixture_rows()
        with tempfile.TemporaryDirectory() as directory, \
                patch("train.make_models", return_value=models), \
                patch("train.select_model", side_effect=record_selection), \
                patch("joblib.dump"), patch("train.plot_confusion"):
            path = Path(directory) / "data.jsonl"
            write_rows(path, rows)
            result = train(path, Path(directory) / "model.joblib", directory, allow_small_prototype=True, profile="baseline")
            saved_dev = json.loads((Path(directory) / "dev_examples.json").read_text(encoding="utf-8"))
        self.assertEqual(events[:6], [("fit", {"train"}, 28), ("predict", {"dev"}, 14)] * 3)
        self.assertEqual(events[6:10], [("select",)] * 4)
        self.assertEqual(events[10:], [("predict", {"test"}, 14)] * 3)
        self.assertEqual(result["selection"]["fit_split"], "train")
        self.assertFalse(result["selection"]["test_used_for_selection"])
        self.assertEqual(saved_dev["partition"], "dev")

    def test_development_profile_rejects_test_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            write_rows(path, fixture_rows())
            with self.assertRaisesRegex(ValueError, "excludes test"):
                read_data(path, min_rows=1)
            with self.assertRaises(ValueError):
                split_data(fixture_rows())

    def test_default_flow_never_predicts_or_scores_test(self):
        events = []

        class Model:
            def fit(self, x, y):
                events.append(("fit", {record["context"].split()[0] for record in x}))
                return self

            def predict(self, x):
                events.append(("predict", {record["context"].split()[0] for record in x}))
                return list(LABELS)

        names = ("multinomial_nb_word", "logistic_regression_word", "linear_svm_word",
                 "logistic_regression_hybrid_structure")
        with tempfile.TemporaryDirectory() as directory, \
                patch("train.make_models", return_value={name: Model() for name in names}) as candidates, \
                patch("joblib.dump"), patch("train.plot_confusion"):
            path = Path(directory) / "development.jsonl"
            write_rows(path, fixture_rows("development"))
            result = train(path, Path(directory) / "model.joblib", directory, allow_small_prototype=True)
            predictions = json.loads((Path(directory) / "predictions.json").read_text(encoding="utf-8"))
            sample = json.loads((Path(directory) / "comparison_set.json").read_text(encoding="utf-8"))
        self.assertEqual(events, [("fit", {"train"}), ("predict", {"dev"})] * 4)
        candidates.assert_called_once_with(42, profile="development")
        self.assertEqual(result["evaluation"]["scope"], "dev_only")
        self.assertEqual(result["evaluation"]["context_margin"], 1000)
        self.assertEqual(result["evaluation"]["test_predictions_per_candidate"], 0)
        self.assertEqual(set(result["partitions"]), {"train", "dev"})
        self.assertTrue(all(model["test"] is None for model in result["models"].values()))
        self.assertEqual(predictions["partition"], "dev")
        self.assertEqual(sample["partition"], "dev")


if __name__ == "__main__":
    unittest.main()
