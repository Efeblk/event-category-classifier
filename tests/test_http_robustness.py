"""Adversarial loopback HTTP checks; fixtures and injected Jev never go online."""

import copy
import io
import json
import socket
import tempfile
import threading
import unittest
from contextlib import contextmanager, redirect_stderr
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from app import MAX_BODY_BYTES, make_handler, validate_metrics
from classifier import ARTIFACT_TASK, ARTIFACT_VERSION, DATASET_DOMAIN, LABELS, make_models, model_input
from method_comparison import METHODS


def matching_metrics(artifacts):
    first = artifacts[METHODS[0]]
    names = {method: artifacts[method]["model_name"] for method in METHODS}
    return {
        "schema_version": ARTIFACT_VERSION, "task": ARTIFACT_TASK,
        "dataset_domain": DATASET_DOMAIN, "seed": first["seed"],
        "code_sha256": first["code_sha256"], "dataset": {"sha256": first["dataset_sha256"]},
        "evaluation": {"scope": "dev_only", "partition": "dev", "context_margin": 1000,
                       "labels": list(LABELS), "test_predictions_per_candidate": 0,
                       "independent_new_test_evaluation": False},
        "selection": {"method_models": names, "selected_model": names["logistic_regression"],
                      "fit_split": "train", "metric": "dev_macro_f1", "test_used_for_selection": False},
        "models": {name: {"dev": {"accuracy": 0.57, "micro_f1": 0.57, "macro_f1": 0.39}, "test": None}
                   for name in names.values()},
    }


@contextmanager
def server_for(artifacts, reports, metrics=None, jev=None):
    errors = io.StringIO()
    server = HTTPServer(("127.0.0.1", 0), make_handler(artifacts, metrics or {}, jev, reports))
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    with redirect_stderr(errors):
        worker.start()
        try:
            yield server.server_address
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
    if errors.getvalue():
        raise AssertionError("The local HTTP server emitted an unhandled exception.")


def exchange(address, body=b"", headers=None, path="/api/compare", method="POST", finish=True):
    if headers is None:
        headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
    wire = f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n".encode("ascii")
    wire += b"".join(f"{key}: {value}\r\n".encode("ascii") for key, value in headers) + b"\r\n" + body
    with socket.create_connection(address, timeout=2) as connection:
        connection.sendall(wire)
        if finish:
            connection.shutdown(socket.SHUT_WR)
        raw = b""
        while True:
            part = connection.recv(65536)
            if not part:
                break
            raw += part
    head, payload = raw.split(b"\r\n\r\n", 1)
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload)


class HttpRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = ("multinomial_nb_word", "logistic_regression_word", "linear_svm_word")
        models = make_models()
        records = [model_input(label.replace("_", " ") + " fixture example") for label in LABELS] * 2
        cls.artifacts = {}
        for method, name in zip(METHODS, names):
            model = models[name]
            model.fit(records, list(LABELS) * 2)
            cls.artifacts[method] = {
                "schema_version": ARTIFACT_VERSION, "task": ARTIFACT_TASK,
                "dataset_domain": DATASET_DOMAIN, "labels": LABELS, "model": model,
                "model_name": name, "seed": 42, "dataset_sha256": "a" * 64,
                "code_sha256": "b" * 64,
                "provenance": {"fit_partition": "train", "evaluation_scope": "dev_only", "context_margin": 1000},
            }

    def test_truncated_body_is_rejected_even_when_shorter_body_is_valid_json(self):
        body = b'{"text":"loaded language"}'
        with tempfile.TemporaryDirectory() as directory, server_for(self.artifacts, directory) as address:
            status, payload = exchange(address, body, [("Content-Type", "application/json"), ("Content-Length", "100")])
        self.assertEqual(status, 400)
        self.assertIn("length", payload["error"])

    def test_incomplete_open_body_times_out_and_server_recovers(self):
        body = b'{"text":"loaded language"}'
        with tempfile.TemporaryDirectory() as directory, patch("app.REQUEST_READ_TIMEOUT_SECONDS", 0.1), \
                server_for(self.artifacts, directory) as address:
            status, payload = exchange(address, body, [("Content-Type", "application/json"), ("Content-Length", "100")], finish=False)
            self.assertEqual(status, 400)
            self.assertIn("timed out", payload["error"])
            self.assertEqual(exchange(address, body)[0], 200)

    def test_ambiguous_lengths_transfer_encoding_and_content_types_are_rejected(self):
        body = b'{"text":"loaded language"}'
        length = ("Content-Length", str(len(body)))
        content_type = ("Content-Type", "application/json")
        cases = [
            [content_type], [content_type, ("Content-Length", "+27")],
            [content_type, ("Content-Length", "-1")],
            [content_type, ("Content-Length", "twenty")],
            [content_type, length, length], [content_type, length, ("Content-Length", "0")],
            [content_type, length, ("Transfer-Encoding", "chunked")],
            [length], [length, ("Content-Type", "text/plain")],
            [length, content_type, content_type],
            [length, ("Content-Type", "application/json; charset=utf-16")],
            [content_type, ("Content-Length", str(MAX_BODY_BYTES + 1))],
        ]
        with tempfile.TemporaryDirectory() as directory, server_for(self.artifacts, directory) as address:
            for headers in cases:
                with self.subTest(headers=headers):
                    status, payload = exchange(address, body, headers)
                    self.assertEqual(status, 400)
                    self.assertIsInstance(payload["error"], str)

    def test_duplicate_members_nonfinite_values_and_invalid_unicode_never_call_models_or_provider(self):
        bodies = [b'{"text":"ignored","text":"loaded language"}',
                  b'{"text":"loaded language","include_jev":false,"include_jev":true}',
                  b'{"text":"loaded language","context":NaN}',
                  b'{"text":"loaded language","context":Infinity}',
                  b'{"text":"loaded language","context":1e9999}',
                  b'{"text":"loaded language\\ud800","include_jev":true}',
                  b'{"text":"loaded language","context":"\\udfff"}', b'\xff']
        bodies.append(b'{"text":' + b'[' * 12000 + b'0' + b']' * 12000 + b'}')
        jev = Mock()
        with tempfile.TemporaryDirectory() as directory, server_for(self.artifacts, directory, jev=jev) as address, \
                patch("app.compare_request") as compare:
            for body in bodies:
                with self.subTest(body=body):
                    status, payload = exchange(address, body)
                    self.assertEqual(status, 400)
                    self.assertIn("error", payload)
        compare.assert_not_called()
        jev.classify.assert_not_called()

    def test_controls_and_invisible_only_inputs_fail_before_prediction_or_provider(self):
        cases = [{"text": "loaded\x00 language"}, {"text": "\u200b\ufeff\u202e"},
                 {"text": "loaded language", "context": "bad\x08context"}]
        jev = Mock()
        with tempfile.TemporaryDirectory() as directory, server_for(self.artifacts, directory, jev=jev) as address, \
                patch("method_comparison.classify_request") as predict:
            for case in cases:
                with self.subTest(case=case):
                    status, payload = exchange(address, json.dumps({**case, "include_jev": True}).encode())
                    self.assertEqual(status, 400)
                    self.assertIn("error", payload)
        predict.assert_not_called()
        jev.classify.assert_not_called()

    def test_codepoint_boundaries_valid_emoji_newlines_and_unrelated_context_are_preserved(self):
        # Context may consist only of nearby sentences; containment is not required.
        text, context = "loaded language " + "\U0001f600" * 4984, "Nearby\n" + "\U0001f600" * 9993
        self.assertEqual(len(text), 5000)
        self.assertEqual(len(context), 10000)
        seen = []
        def compare(artifacts, excerpt, jev, include_jev, surrounding):
            from method_comparison import validate_input
            validate_input(excerpt, surrounding, include_jev)
            seen.append((excerpt, surrounding))
            return {"results": []}
        with tempfile.TemporaryDirectory() as directory, server_for(self.artifacts, directory) as address, \
                patch("app.compare_request", side_effect=compare):
            body = json.dumps({"text": text, "context": context}).encode()
            self.assertLess(len(body), MAX_BODY_BYTES)
            self.assertEqual(exchange(address, body)[0], 200)
            for oversized in ({"text": text + "\U0001f600"}, {"text": "loaded language", "context": context + "x"}):
                self.assertEqual(exchange(address, json.dumps(oversized).encode())[0], 400)
        self.assertEqual(seen, [(text, context)])

    def test_populated_partial_stale_and_nonfinite_metrics_are_rejected(self):
        valid = matching_metrics(self.artifacts)
        self.assertIs(validate_metrics(valid, self.artifacts), valid)
        self.assertEqual(validate_metrics({}, self.artifacts), {})
        mutations = [lambda report: report.update(code_sha256="c" * 64),
                     lambda report: report["dataset"].update(sha256="c" * 64),
                     lambda report: report.update(seed=43),
                     lambda report: report["evaluation"].update(scope="historical_baseline"),
                     lambda report: report["selection"]["method_models"].update(logistic_regression="other_model"),
                     lambda report: report["models"][next(iter(report["models"]))].update(test={"accuracy": 0.7}),
                     lambda report: report["models"][next(iter(report["models"]))]["dev"].update(accuracy=float("nan")),
                     lambda report: report["models"][next(iter(report["models"]))]["dev"].update(accuracy=10 ** 400),
                     lambda report: report.update(note="invalid\ud800"),
                     lambda report: report.update(note=float("inf"))]
        with self.assertRaises(ValueError):
            make_handler(self.artifacts, {"models": {"stale": {"dev": {"accuracy": 0.99}}}})
        for mutate in mutations:
            report = copy.deepcopy(valid)
            mutate(report)
            with self.subTest(report=report), self.assertRaises(ValueError):
                make_handler(self.artifacts, report)

    def test_cached_reports_must_identify_the_exact_artifact_run_and_context(self):
        provenance = {"training_data_sha256": "a" * 64, "training_code_sha256": "b" * 64,
                      "training_seed": 42, "model_evaluation_scope": "dev_only", "context_margin": 1000}
        examples = {**provenance, "partition": "dev", "records": [
            {"id": "fixture", "label": LABELS[8], "text": "loaded language", "context": "Nearby sentences."}]}
        comparison = {**provenance, "task": ARTIFACT_TASK, "partition": "dev", "labels": list(LABELS),
                      "method_models": {method: artifact["model_name"] for method, artifact in self.artifacts.items()},
                      "examples": 1, "methods": {method: {"completed": 1, "total": 1, "accuracy": 0.5, "macro_f1": 0.39}
                                                for method in METHODS}}
        comparison["methods"]["jev"] = {"completed": 0, "total": 1, "accuracy": None, "macro_f1": None}
        with tempfile.TemporaryDirectory() as directory, server_for(self.artifacts, directory) as address:
            path = Path(directory)
            for name, endpoint, report in (("dev_examples.json", "/api/examples", examples),
                                           ("method_comparison.json", "/api/method-results", comparison)):
                (path / name).write_text(json.dumps(report), encoding="utf-8")
                self.assertEqual(exchange(address, path=endpoint, method="GET"), (200, report))
                for key, value in (("training_data_sha256", "c" * 64), ("training_code_sha256", "c" * 64),
                                   ("training_seed", 43), ("context_margin", 350)):
                    with self.subTest(name=name, key=key):
                        stale = {**report, key: value}
                        (path / name).write_text(json.dumps(stale), encoding="utf-8")
                        status, payload = exchange(address, path=endpoint, method="GET")
                        self.assertEqual(status, 200)
                        self.assertFalse(payload.get("records", payload.get("methods")))
            comparison["methods"]["jev"]["accuracy"] = 0.8
            (path / "method_comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
            self.assertEqual(exchange(address, path="/api/method-results", method="GET"), (200, {}))

    def test_invalid_cached_examples_and_method_reports_become_unavailable(self):
        with tempfile.TemporaryDirectory() as directory, server_for(self.artifacts, directory) as address:
            path = Path(directory)
            for report in ({"partition": "test", "records": []},
                           {"partition": "dev", "records": [{"id": "bad", "text": "hi", "label": "invented"}]}):
                (path / "dev_examples.json").write_text(json.dumps(report), encoding="utf-8")
                status, payload = exchange(address, path="/api/examples", method="GET")
                self.assertEqual(status, 200)
                self.assertEqual(payload["records"], [])
                self.assertEqual(payload["partition"], "dev")
            for body in ('{"methods":{},"training_data_sha256":"stale"}', '{"methods":{"x":NaN}}', '{"context":"\\ud800"}'):
                (path / "method_comparison.json").write_text(body, encoding="utf-8")
                self.assertEqual(exchange(address, path="/api/method-results", method="GET"), (200, {}))


if __name__ == "__main__":
    unittest.main()
