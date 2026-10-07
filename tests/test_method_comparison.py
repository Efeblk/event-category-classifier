import json
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import make_handler
from classifier import ARTIFACT_TASK, DATASET_DOMAIN, LABELS, OUTPUT_LABELS, make_models
from compare_methods import evaluate
from jev import JevClient, load_settings, validate_jev_response
from method_comparison import compare_request


def response(label="calendar_set"):
    return {"model": "jev-1.13.0", "answers": {"activity": {
        "type": "choice", "choice": label, "confidence": 0.8,
        "probabilities": {key: 0.85 if key == label else 0.15 / (len(OUTPUT_LABELS)-1) for key in OUTPUT_LABELS}}},
        "usage": {"input_tokens": 100, "output_tokens": 20}}


class MethodComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = make_models()["logistic_regression_char"]
        examples = ["concert live music", "concert tickets", "theatre stage play", "theatre actors",
                    "standup comedian comedy", "standup comedy show"]
        model.fit([label.replace("_", " ")+" fixture" for label in LABELS]*2, list(LABELS)*2)
        cls.artifact = {"model": model, "model_name": "logistic_regression_char", "labels": LABELS,
                        "dataset_sha256": "test", "seed": 42, "task": ARTIFACT_TASK,
                        "dataset_domain": DATASET_DOMAIN, "provenance": "test"}

    def test_methods_receive_identical_original_text_without_lr_language_guards(self):
        text = "Konser değil, tiyatro istiyorum."
        jev = Mock()
        jev.classify.return_value = {"status": "ok", "label": "calendar_set"}
        with patch("method_comparison.parse_request", return_value="calendar_set") as rules, \
             patch("method_comparison.classify_request", return_value={"label": "play_music"}) as learned:
            result = compare_request(self.artifact, text, jev, True)
        rules.assert_called_once_with(text)
        learned.assert_called_once_with(self.artifact, text)
        jev.classify.assert_called_once_with(text)
        self.assertEqual([row["label"] for row in result["results"]], ["calendar_set", "play_music", "calendar_set"])

    def test_optional_jev_is_not_called_by_default(self):
        jev = Mock()
        result = compare_request(self.artifact, "concert tickets", jev)
        jev.classify.assert_not_called()
        self.assertEqual(result["results"][2]["status"], "not_requested")
        self.assertNotIn("label", result["results"][2])

    def test_unconfigured_jev_has_no_fake_prediction(self):
        result = compare_request(self.artifact, "concert tickets", include_jev=True)
        self.assertEqual(result["results"][2]["status"], "not_configured")
        self.assertNotIn("label", result["results"][2])

    def test_invalid_input_is_rejected_before_provider_calls(self):
        jev = Mock()
        for text in (None, " ", "x" * 5001):
            with self.assertRaises(ValueError):
                compare_request(self.artifact, text, jev, True)
        with self.assertRaises(ValueError):
            compare_request(self.artifact, "play_music", jev, "true")
        jev.classify.assert_not_called()

    def test_http_compare_returns_three_results_and_keeps_configuration_private(self):
        with tempfile.TemporaryDirectory() as directory:
            jev = JevClient({"TYPESAFE_API_KEY": "private-test-key", "JEV_MAX_CALLS": "1"},
                            directory, Mock(return_value=response()))
            server = HTTPServer(("127.0.0.1", 0), make_handler(self.artifact, {}, jev))
            worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
            worker.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/comparison-config", timeout=2) as handle:
                    public = handle.read()
                self.assertNotIn(b"private-test-key", public)
                body = json.dumps({"text": "theatre play", "include_jev": True}).encode()
                with urlopen(Request(base + "/api/compare", data=body,
                                     headers={"Content-Type": "application/json"}), timeout=2) as handle:
                    result = json.load(handle)
                self.assertEqual([row["method"] for row in result["results"]],
                                 ["parser", "logistic_regression", "jev"])
                self.assertEqual(result["results"][2]["label"], "calendar_set")
                bad = json.dumps({"text": "theatre play", "include_jev": "true"}).encode()
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + "/api/compare", data=bad), timeout=2)
                self.assertEqual(error.exception.code, 400)
                self.assertEqual(jev.transport.call_count, 1)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)

    def test_saved_experiment_uses_same_cases_and_gives_missing_jev_no_score(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "cases.json"
            output_path = Path(directory) / "result.json"
            data_path.write_text(json.dumps({"provenance": "test fixture", "records": [
                {"text": "concert tickets", "label": "play_music"},
                {"text": "theatre play", "label": "calendar_set"}]}), encoding="utf-8")
            with patch("compare_methods.joblib.load", return_value=self.artifact), \
                 patch("compare_methods.JevClient", return_value=JevClient({}, directory)):
                result = evaluate(Path("unused.joblib"), data_path, output_path)
            self.assertEqual(len(result["records"]), 2)
            self.assertEqual(result["methods"]["parser"]["completed"], 2)
            self.assertEqual(result["methods"]["jev"]["not_run"], 2)
            self.assertIsNone(result["methods"]["jev"]["accuracy"])
            self.assertIsNone(result["methods"]["jev"]["macro_f1"])
            self.assertEqual(result["task"], ARTIFACT_TASK)

    def test_abstaining_counts_as_wrong_and_unused_unclear_label_is_not_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "cases.json"
            data_path.write_text(json.dumps({"records": [
                {"text": "concert tickets", "label": "play_music"},
                {"text": "theatre play", "label": "calendar_set"}]}), encoding="utf-8")
            with patch("compare_methods.joblib.load", return_value=self.artifact),                  patch("compare_methods.JevClient", return_value=JevClient({}, directory)),                  patch("method_comparison.parse_request", side_effect=["play_music", "unclear"]):
                result = evaluate(Path("unused.joblib"), data_path, Path(directory) / "result.json")
            parser = result["methods"]["parser"]
            self.assertEqual(parser["accuracy"], 0.5)
            self.assertNotIn("unclear", parser["per_class"])
            self.assertEqual(parser["macro_f1"], 0.5)

    def test_partial_provider_run_has_no_comparable_accuracy_score(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "cases.json"
            data_path.write_text(json.dumps({"records": [
                {"text": "theatre play", "label": "calendar_set"},
                {"text": "concert tickets", "label": "play_music"}]}), encoding="utf-8")
            client = Mock()
            client.configured = True
            client.remaining_calls.return_value = 2
            client.model = "jev-1.13.0"
            client.classify.side_effect = [{"status": "ok", "label": "calendar_set"},
                                          {"status": "error", "message": "Unavailable"}]
            with patch("compare_methods.joblib.load", return_value=self.artifact), \
                 patch("compare_methods.JevClient", return_value=client):
                result = evaluate(Path("unused.joblib"), data_path, Path(directory) / "result.json", True)
            self.assertEqual(result["methods"]["jev"]["completed"], 1)
            self.assertEqual(result["methods"]["jev"]["failed"], 1)
            self.assertIsNone(result["methods"]["jev"]["accuracy"])


class JevClientTests(unittest.TestCase):
    def test_valid_provider_answer_is_accepted(self):
        self.assertEqual(validate_jev_response(response(), "jev-1.13.0")["choice"], "calendar_set")

    def test_malformed_provider_answers_are_rejected(self):
        for mutate in (
            lambda data: data.update(model="jev-other"),
            lambda data: data.update(answers=[]),
            lambda data: data["answers"]["activity"].update(choice="movie"),
            lambda data: data["answers"]["activity"]["probabilities"].update(concert=float("nan")),
            lambda data: data["answers"]["activity"].update(confidence=True),
            lambda data: data["usage"].update(input_tokens=-1),
        ):
            data = response()
            mutate(data)
            with self.assertRaises(ValueError):
                validate_jev_response(data, "jev-1.13.0")

    def test_budget_is_shared_across_clients_and_survives_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = {"TYPESAFE_API_KEY": "test-key", "JEV_MAX_CALLS": "1"}
            transport = Mock(side_effect=TimeoutError("private provider details"))
            client = JevClient(settings, directory, transport)
            result = client.classify("a theatre play")
            self.assertEqual(result["status"], "error")
            self.assertNotIn("private", result["message"])
            self.assertEqual(transport.call_count, 1)
            second = JevClient(settings, directory, Mock(return_value=response()))
            self.assertEqual(second.classify("a theatre play")["status"], "budget_exhausted")
            second.transport.assert_not_called()
            receipt = json.loads((Path(directory) / "001.json").read_text())
            self.assertEqual(receipt["status"], "error")

    def test_live_request_has_same_text_and_four_labels_but_no_key_in_body_or_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = response()
            provider["debug"] = "private-test-key"
            provider["usage"]["debug"] = "private-test-key"
            transport = Mock(return_value=provider)
            client = JevClient({"TYPESAFE_API_KEY": "private-test-key", "JEV_MAX_CALLS": "1"}, directory, transport)
            result = client.classify("Konser değil, tiyatro istiyorum.")
            self.assertEqual(result["label"], "calendar_set")
            self.assertNotIn("private-test-key", json.dumps(result))
            body = transport.call_args.args[0]
            payload = json.loads(body)
            self.assertEqual(payload["state"]["text"], "Konser değil, tiyatro istiyorum.")
            self.assertEqual(set(payload["questions"]["activity"]["criteria"]), set(OUTPUT_LABELS))
            self.assertNotIn(b"private-test-key", body)
            receipt = (Path(directory) / "001.json").read_text()
            self.assertNotIn("private-test-key", receipt)
            self.assertNotIn("Konser", receipt)

    def test_disabled_and_oversized_requests_make_no_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = Mock()
            disabled = JevClient({}, directory, transport)
            self.assertEqual(disabled.classify("play_music")["status"], "not_configured")
            enabled = JevClient({"TYPESAFE_API_KEY": "test", "JEV_MAX_CALLS": "1"}, directory, transport)
            self.assertEqual(enabled.classify("x" * 1001)["status"], "not_run")
            self.assertEqual(enabled.remaining_calls(), 1)
            transport.assert_not_called()

    def test_settings_keep_credentials_server_side_and_environment_takes_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("TYPESAFE_API_KEY=local-secret\nJEV_MAX_CALLS=5\n", encoding="utf-8")
            settings = load_settings(path, {"JEV_MAX_CALLS": "2"})
            client = JevClient(settings, directory)
            self.assertEqual(client.max_calls, 2)
            self.assertNotIn("local-secret", json.dumps(client.public_status()))

    def test_invalid_limits_and_unpinned_models_are_rejected(self):
        for settings in ({"JEV_MAX_CALLS": "-1"}, {"JEV_MAX_CALLS": "51"},
                         {"JEV_MAX_CALLS": "many"}, {"JEV_MODEL": "jev-latest"}):
            with self.assertRaises(ValueError):
                JevClient(settings)


if __name__ == "__main__":
    unittest.main()
