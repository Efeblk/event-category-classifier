"""Local comparison and HTTP checks; provider transports never use the network."""

import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import MAX_BODY_BYTES, make_handler
from classifier import ARTIFACT_TASK, ARTIFACT_VERSION, DATASET_DOMAIN, LABELS, make_models, model_input
from compare_methods import evaluate
from jev import JevClient
from method_comparison import METHODS, compare_request, validate_artifacts


@contextmanager
def running_server(artifacts, directory, jev=None, metrics=None, baseline_path=None):
    server = HTTPServer(("127.0.0.1", 0), make_handler(artifacts, metrics or {}, jev, directory, baseline_path))
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class MethodComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = make_models()
        names = ("multinomial_nb_word", "logistic_regression_word", "linear_svm_word")
        records = [model_input(label.replace("_", " ") + " classification fixture") for label in LABELS] * 2
        cls.artifacts = {}
        for method, name in zip(METHODS, names):
            model = candidates[name]
            model.fit(records, list(LABELS) * 2)
            cls.artifacts[method] = {
                "schema_version": ARTIFACT_VERSION,
                "model": model,
                "model_name": name,
                "labels": LABELS,
                "dataset_sha256": "a" * 64,
                "code_sha256": "b" * 64,
                "seed": 42,
                "task": ARTIFACT_TASK,
                "dataset_domain": DATASET_DOMAIN,
                "provenance": {"fit_partition": "train", "source": "test fixture", "context_margin": 1000, "evaluation_scope": "dev_only"},
            }

    def test_all_methods_receive_identical_original_excerpt_and_context(self):
        text, context = "Emotionally charged words! 😀", "Surrounding context, unchanged."
        jev = Mock()
        jev.classify.return_value = {"status": "ok", "label": LABELS[3]}
        with patch("method_comparison.classify_request", side_effect=[{"label": label} for label in LABELS[:3]]) as learned:
            result = compare_request(self.artifacts, text, jev, True, context)
        self.assertEqual(learned.call_count, 3)
        for call, method in zip(learned.call_args_list, METHODS):
            self.assertIs(call.args[0], self.artifacts[method])
            self.assertEqual(call.args[1], text)
            self.assertEqual(call.kwargs, {"context": context})
        jev.classify.assert_called_once_with(text, context=context)
        self.assertEqual([row["method"] for row in result["results"]], [*METHODS, "jev"])
        self.assertEqual([row["label"] for row in result["results"]], list(LABELS[:4]))

    def test_provider_is_not_called_by_default_or_given_a_fake_prediction(self):
        jev = Mock()
        result = compare_request(self.artifacts, "loaded language", jev)
        jev.classify.assert_not_called()
        self.assertEqual(result["results"][-1]["status"], "not_requested")
        self.assertNotIn("label", result["results"][-1])
        unconfigured = compare_request(self.artifacts, "loaded language", include_jev=True)
        self.assertEqual(unconfigured["results"][-1]["status"], "not_configured")
        self.assertNotIn("label", unconfigured["results"][-1])

    def test_invalid_fields_are_rejected_before_any_provider_attempt(self):
        jev = Mock()
        for text in (None, " ", "x" * 5001, ["text"]):
            with self.subTest(text_type=type(text).__name__), self.assertRaises(ValueError):
                compare_request(self.artifacts, text, jev, True)
        for context in (None, 12, "x" * 10001):
            with self.subTest(context_type=type(context).__name__), self.assertRaises(ValueError):
                compare_request(self.artifacts, "loaded language", jev, True, context)
        for include in (1, "true", None):
            with self.assertRaises(ValueError):
                compare_request(self.artifacts, "loaded language", jev, include)
        jev.classify.assert_not_called()

    def test_method_family_and_training_source_must_match(self):
        with self.assertRaises(ValueError):
            validate_artifacts({"logistic_regression": self.artifacts["logistic_regression"]})
        swapped = dict(self.artifacts)
        swapped["naive_bayes"] = swapped["linear_svm"]
        with self.assertRaisesRegex(ValueError, "wrong model family"):
            validate_artifacts(swapped)
        for field, value in (("dataset_sha256", "c" * 64), ("code_sha256", "c" * 64), ("seed", 99)):
            mismatched = dict(self.artifacts)
            mismatched["linear_svm"] = {**mismatched["linear_svm"], field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "same training"):
                validate_artifacts(mismatched)
        mixed_scope = dict(self.artifacts)
        mixed_scope["linear_svm"] = {
            **mixed_scope["linear_svm"],
            "provenance": {"fit_partition": "train", "context_margin": 350, "evaluation_scope": "historical_baseline"},
        }
        with self.assertRaisesRegex(ValueError, "same provenance"):
            validate_artifacts(mixed_scope)

    def test_outside_vocabulary_abstains_independently_for_all_local_models(self):
        result = compare_request(self.artifacts, "zzzzzzzzzz")
        for row in result["results"][:3]:
            self.assertEqual(row["label"], "unclear")
            self.assertEqual(row["reason"], "unknown_terms")

    def test_http_returns_four_methods_actual_examples_and_private_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            examples = {"partition": "dev", "records": [{"id": "dev-1", "text": "loaded language", "context": "", "label": LABELS[8]}]}
            examples.update(training_data_sha256="a" * 64, training_code_sha256="b" * 64,
                            training_seed=42, model_evaluation_scope="dev_only", context_margin=1000)
            Path(directory, "dev_examples.json").write_text(json.dumps(examples), encoding="utf-8")
            client = JevClient({"TYPESAFE_API_KEY": "private-test-key", "JEV_MAX_CALLS": "0"}, directory, Mock())
            with running_server(self.artifacts, directory, client) as base:
                with urlopen(base + "/api/comparison-config", timeout=2) as handle:
                    self.assertNotIn(b"private-test-key", handle.read())
                with urlopen(base + "/api/examples", timeout=2) as handle:
                    self.assertEqual(json.load(handle), examples)
                body = json.dumps({"text": "loaded language", "context": "actual context"}).encode()
                with urlopen(Request(base + "/api/compare", data=body, headers={"Content-Type": "application/json"}), timeout=2) as handle:
                    result = json.load(handle)
                    self.assertEqual(handle.headers["Cache-Control"], "no-store")
                self.assertEqual([row["method"] for row in result["results"]], [*METHODS, "jev"])
                self.assertEqual(result["results"][-1]["status"], "not_requested")
                client.transport.assert_not_called()

    def test_http_bad_json_unknown_fields_oversized_body_and_wrong_content_type_are_400(self):
        with tempfile.TemporaryDirectory() as directory, running_server(self.artifacts, directory) as base:
            bad_requests = [
                (b"not json", {}),
                (b"\xff", {}),
                (b"[]", {}),
                (json.dumps({"text": "valid", "include_jev": 1}).encode(), {}),
                (json.dumps({"text": "valid", "context": None}).encode(), {}),
                (json.dumps({"text": "valid", "secret": "ignored?"}).encode(), {}),
                (b"x", {"Content-Length": str(MAX_BODY_BYTES + 1)}),
                (json.dumps({"text": "valid"}).encode(), {"Content-Type": "text/plain"}),
            ]
            for data, extra in bad_requests:
                with self.subTest(data=data[:30], headers=extra), self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + "/api/compare", data=data, headers={"Content-Type": "application/json", **extra}), timeout=2)
                self.assertEqual(error.exception.code, 400)
                with error.exception as handle:
                    self.assertIn("error", json.load(handle))

    def test_saved_sample_keeps_ids_context_and_scores_all_fourteen_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory, "cases.json")
            rows = [{"id": "one", "text": "loaded language", "context": "context one", "label": LABELS[8]},
                    {"id": "two", "text": "slogans", "context": "context two", "label": LABELS[11]}]
            data_path.write_text(json.dumps({"partition": "dev", "provenance": "test fixture", "records": rows}), encoding="utf-8")
            def predictions(artifacts, text, jev, include, context):
                label = LABELS[8] if text == "loaded language" else "unclear"
                return {"results": [{"method": method, "status": "ok", "label": label} for method in METHODS]
                                   + [{"method": "jev", "status": "not_requested"}]}
            with patch("compare_methods.joblib.load", return_value=self.artifacts), \
                 patch("compare_methods.JevClient", return_value=JevClient({}, directory)), \
                 patch("compare_methods.compare_request", side_effect=predictions) as compare:
                result = evaluate(Path("unused.joblib"), data_path, Path(directory, "result.json"))
            self.assertEqual(compare.call_args_list[0].args[-1], "context one")
            self.assertEqual(compare.call_args_list[1].args[-1], "context two")
            self.assertEqual(result["records"][0]["id"], "one")
            self.assertEqual(result["partition"], "dev")
            self.assertEqual(result["model_evaluation_scope"], "dev_only")
            self.assertIn("Sample partition: dev", result["note"])
            self.assertNotIn("held-out test", result["note"])
            self.assertEqual(result["methods"]["linear_svm"]["accuracy"], 0.5)
            self.assertAlmostEqual(result["methods"]["linear_svm"]["macro_f1"], 1 / len(LABELS))
            self.assertEqual(len([label for label in LABELS if label in result["methods"]["linear_svm"]["per_class"]]), 14)
            self.assertIsNone(result["methods"]["jev"]["accuracy"])
            self.assertIsNone(result["methods"]["jev"]["macro_f1"])

    def test_partial_provider_run_keeps_comparable_scores_null(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory, "cases.json")
            data_path.write_text(json.dumps({"partition": "dev", "records": [
                {"id": "one", "text": "loaded language", "context": "", "label": LABELS[8]},
                {"id": "two", "text": "slogans", "context": "", "label": LABELS[11]},
            ]}), encoding="utf-8")
            client = Mock()
            client.configured = True
            client.remaining_calls.return_value = 2
            client.model = "jev-1.13.0"
            client.classify.side_effect = [{"status": "ok", "label": LABELS[8]}, {"status": "error", "message": "Unavailable"}]
            with patch("compare_methods.joblib.load", return_value=self.artifacts), patch("compare_methods.JevClient", return_value=client):
                result = evaluate(Path("unused.joblib"), data_path, Path(directory, "result.json"), True)
            self.assertEqual(result["methods"]["jev"]["completed"], 1)
            self.assertEqual(result["methods"]["jev"]["failed"], 1)
            self.assertIsNone(result["methods"]["jev"]["accuracy"])
            self.assertIsNone(result["methods"]["jev"]["macro_f1"])

    def test_invalid_dataset_is_rejected_before_loading_models_or_creating_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory, "cases.json")
            for rows in ([{"id": "one", "text": "valid", "label": "unclear"}],
                         [{"id": "one", "text": "valid", "label": LABELS[0]}] * 2):
                data_path.write_text(json.dumps({"records": rows}), encoding="utf-8")
                with patch("compare_methods.JevClient") as provider, patch("compare_methods.joblib.load") as load, self.assertRaises(ValueError):
                    evaluate(Path("unused.joblib"), data_path, Path(directory, "result.json"), True)
                provider.assert_not_called()
                load.assert_not_called()

    def test_every_provider_payload_is_validated_before_first_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory, "cases.json")
            data_path.write_text(json.dumps({"partition": "dev", "records": [
                {"id": "one", "text": "loaded language", "context": "", "label": LABELS[8]},
                {"id": "two", "text": "slogans", "context": "", "label": LABELS[11]},
            ]}), encoding="utf-8")
            client = Mock()
            client.prepare_request.side_effect = [b"valid payload", ValueError("Provider payload is too large.")]
            with patch("compare_methods.joblib.load", return_value=self.artifacts), patch("compare_methods.JevClient", return_value=client), self.assertRaisesRegex(ValueError, "too large"):
                evaluate(Path("unused.joblib"), data_path, Path(directory, "result.json"), True)
            self.assertEqual(client.prepare_request.call_count, 2)
            client.classify.assert_not_called()

    def test_current_dev_results_and_historical_test_results_stay_separate(self):
        from tests.test_http_robustness import matching_metrics
        current = matching_metrics(self.artifacts)
        historical = {"models": {"previous": {"dev": {"accuracy": 0.49}, "test": {"accuracy": 0.47}}}}
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory, "baseline.json")
            baseline_path.write_text(json.dumps(historical), encoding="utf-8")
            with running_server(self.artifacts, directory, metrics=current, baseline_path=baseline_path) as base:
                with urlopen(base + "/api/results", timeout=2) as handle:
                    actual = json.load(handle)
                self.assertEqual(actual, current)
                self.assertIsNone(actual["models"][self.artifacts["logistic_regression"]["model_name"]]["test"])
                with urlopen(base + "/api/baseline-results", timeout=2) as handle:
                    self.assertEqual(json.load(handle), historical)

    def test_development_artifacts_reject_non_dev_samples_before_prediction_or_provider_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory, "cases.json")
            for partition in ("test", "train", "unspecified"):
                data_path.write_text(json.dumps({"partition": partition, "records": [
                    {"id": "fixture", "text": "loaded language", "context": "", "label": LABELS[8]},
                ]}), encoding="utf-8")
                with patch("compare_methods.joblib.load", return_value=self.artifacts), \
                     patch("compare_methods.JevClient") as provider, \
                     patch("compare_methods.compare_request") as predict, \
                     self.subTest(partition=partition), self.assertRaisesRegex(ValueError, "require a dev sample"):
                    evaluate(Path("unused.joblib"), data_path, Path(directory, "result.json"), True)
                provider.assert_not_called()
                predict.assert_not_called()

    def test_development_demo_rejects_historical_artifacts(self):
        historical = {
            method: {**artifact, "provenance": {"fit_partition": "train", "context_margin": 350, "evaluation_scope": "historical_baseline"}}
            for method, artifact in self.artifacts.items()
        }
        with self.assertRaisesRegex(ValueError, "development-only demo"):
            make_handler(historical, {"evaluation": {"scope": "dev_only"}})


if __name__ == "__main__":
    unittest.main()
