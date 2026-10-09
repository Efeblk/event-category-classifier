"""Serve the technique-classification course demo on 127.0.0.1 only."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib

from jev import JevClient
from json_validation import strict_json
from classifier import ARTIFACT_TASK, ARTIFACT_VERSION, DATASET_DOMAIN, LABELS
from method_comparison import METHODS, compare_request, validate_artifacts, validate_input


ROOT = Path(__file__).resolve().parent
MAX_BODY_BYTES = 200_000  # Allows 15,000 escaped Unicode code points in JSON.
REQUEST_READ_TIMEOUT_SECONDS = 5.0


def valid_score(value):
    return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)


def validate_metrics(metrics, artifacts):
    """Do not display scores from a missing, partial or different training run."""
    if metrics == {}:
        return metrics
    first = artifacts[METHODS[0]]
    if not isinstance(metrics, dict):
        raise ValueError("Current metrics must be a JSON object.")
    try:
        json.dumps(metrics, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Current metrics must contain finite values and valid UTF-8 JSON.") from error
    expected = {"schema_version": ARTIFACT_VERSION, "task": ARTIFACT_TASK,
                "dataset_domain": DATASET_DOMAIN, "seed": first["seed"],
                "code_sha256": first["code_sha256"]}
    if (type(metrics.get("seed")) is not int or type(metrics.get("schema_version")) is not int
            or any(metrics.get(key) != value for key, value in expected.items())):
        raise ValueError("Current metrics do not match the loaded model run; rerun train.py.")
    dataset, evaluation, selection = (metrics.get(key) for key in ("dataset", "evaluation", "selection"))
    models = metrics.get("models")
    method_models = {method: artifacts[method]["model_name"] for method in METHODS}
    if not isinstance(dataset, dict) or dataset.get("sha256") != first["dataset_sha256"]:
        raise ValueError("Current metrics do not match the loaded training data; rerun train.py.")
    if (not isinstance(evaluation, dict) or evaluation.get("scope") != "dev_only"
            or evaluation.get("partition") != "dev" or evaluation.get("context_margin") != 1000
            or evaluation.get("labels") != list(LABELS)
            or evaluation.get("test_predictions_per_candidate") != 0
            or evaluation.get("independent_new_test_evaluation") is not False):
        raise ValueError("Current metrics must identify the development-only evaluation.")
    if (not isinstance(selection, dict) or selection.get("method_models") != method_models
            or selection.get("fit_split") != "train" or selection.get("metric") != "dev_macro_f1"
            or selection.get("test_used_for_selection") is not False):
        raise ValueError("Current metrics select different models than the loaded artifacts; rerun train.py.")
    if not isinstance(models, dict) or not models or not set(method_models.values()).issubset(models):
        raise ValueError("Current metrics need every loaded model's development scores.")
    if not isinstance(selection.get("selected_model"), str) or selection["selected_model"] not in models:
        raise ValueError("Current metrics need scores for the selected overall model.")
    for result in models.values():
        if not isinstance(result, dict) or result.get("test", "missing") is not None:
            raise ValueError("Current metrics must leave all test scores null.")
        dev = result.get("dev")
        if not isinstance(dev, dict) or not all(valid_score(dev.get(key)) for key in ("accuracy", "micro_f1", "macro_f1")):
            raise ValueError("Current metrics need finite development scores between zero and one.")
    return metrics


def validate_report_provenance(report, artifacts):
    first = artifacts[METHODS[0]]
    expected = {"training_data_sha256": first["dataset_sha256"],
                "training_code_sha256": first["code_sha256"], "training_seed": first["seed"],
                "model_evaluation_scope": "dev_only", "context_margin": 1000}
    if not isinstance(report, dict) or type(report.get("training_seed")) is not int or any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Cached report does not match the loaded development model run.")


def validate_examples(report, artifacts):
    validate_report_provenance(report, artifacts)
    if not isinstance(report, dict) or report.get("partition") != "dev" or not isinstance(report.get("records"), list):
        raise ValueError("Demo examples must identify the development partition.")
    ids = set()
    for row in report["records"]:
        if not isinstance(row, dict) or row.get("label") not in LABELS:
            raise ValueError("Demo examples need supported technique labels.")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
            raise ValueError("Demo examples need unique, non-empty string IDs.")
        ids.add(identifier)
        validate_input(row.get("text"), row.get("context", ""))
    return report


def validate_method_report(report, artifacts):
    if report == {}:
        return report
    validate_report_provenance(report, artifacts)
    first = artifacts[METHODS[0]]
    if (not isinstance(report, dict) or report.get("task") != ARTIFACT_TASK
            or report.get("partition") != "dev" or report.get("model_evaluation_scope") != "dev_only"
            or report.get("training_data_sha256") != first["dataset_sha256"]
            or report.get("method_models") != {method: artifacts[method]["model_name"] for method in METHODS}
            or report.get("labels") != list(LABELS)):
        raise ValueError("Method comparison report does not match the loaded development models.")
    count, methods = report.get("examples"), report.get("methods")
    if type(count) is not int or count <= 0:
        raise ValueError("Method comparison report needs a positive example count.")
    if not isinstance(methods, dict) or set(methods) != {*METHODS, "jev"}:
        raise ValueError("Method comparison report needs all four method entries.")
    for result in methods.values():
        if not isinstance(result, dict):
            raise ValueError("Method comparison scores must be JSON objects.")
        if type(result.get("completed")) is not int or not 0 <= result["completed"] <= count or result.get("total") != count:
            raise ValueError("Method comparison report has invalid completion counts.")
        complete = result["completed"] == count
        for key in ("accuracy", "macro_f1"):
            if (complete and not valid_score(result.get(key))) or (not complete and result.get(key) is not None):
                raise ValueError("Method comparison scores need complete, finite evaluations or null values.")
    return report


def make_handler(artifacts, metrics, jev=None, reports_dir=None, baseline_path=None):
    validate_artifacts(artifacts)
    if any(artifact["provenance"]["evaluation_scope"] != "dev_only" for artifact in artifacts.values()):
        raise ValueError("This development-only demo requires dev-only artifacts.")
    validate_metrics(metrics, artifacts)
    reports_dir = Path(reports_dir) if reports_dir is not None else ROOT / "reports"
    baseline_path = Path(baseline_path) if baseline_path is not None else ROOT / "evidence/baseline/metrics.json"

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(REQUEST_READ_TIMEOUT_SECONDS)

        def read_body(self, length):
            """Bound total upload time even when a client keeps sending bytes."""
            deadline = time.monotonic() + REQUEST_READ_TIMEOUT_SECONDS
            body = bytearray()
            try:
                while len(body) < length:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Request body deadline expired.")
                    self.connection.settimeout(remaining)
                    # read1 makes at most one socket read, so frequent small chunks
                    # cannot restart the overall deadline inside BufferedReader.read.
                    chunk = self.rfile.read1(min(length - len(body), 65536))
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Request body deadline expired.")
                    if not chunk:
                        break
                    body.extend(chunk)
            finally:
                self.connection.settimeout(REQUEST_READ_TIMEOUT_SECONDS)
            if len(body) != length:
                raise ValueError("Request body length does not match Content-Length.")
            return bytes(body)

        def respond(self, status, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # A disconnected browser must not produce a request-body log.

        def report(self, name, default):
            return self.read_report(reports_dir / name, default)

        def read_report(self, path, default, validator=None):
            try:
                report = strict_json(path.read_bytes()) if path.exists() else default
                return validator(report) if validator else report
            except (ValueError, OSError):
                return default

        def do_GET(self):
            if self.path == "/":
                self.respond(200, (ROOT / "demo.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/results":
                self.respond(200, metrics)
            elif self.path == "/api/baseline-results":
                self.respond(200, self.read_report(baseline_path, {}))
            elif self.path == "/api/examples":
                default = {"partition": "dev", "records": [], "message": "Train first to generate valid dev examples."}
                self.respond(200, self.read_report(reports_dir / "dev_examples.json", default,
                                                  lambda report: validate_examples(report, artifacts)))
            elif self.path == "/api/comparison-config":
                self.respond(200, jev.public_status() if jev else {"configured": False, "remaining_calls": 0})
            elif self.path == "/api/method-results":
                self.respond(200, self.read_report(reports_dir / "method_comparison.json", {},
                                                  lambda report: validate_method_report(report, artifacts)))
            else:
                self.respond(404, {"error": "Page not found."})

        def do_POST(self):
            if self.path != "/api/compare":
                return self.respond(404, {"error": "Page not found."})
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0].strip(" \t")):
                    raise ValueError("Send exactly one valid Content-Length header.")
                if self.headers.get_all("Transfer-Encoding"):
                    raise ValueError("Transfer-Encoding is not supported; send a fixed-length JSON body.")
                length = int(lengths[0])
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("Request is too large or empty.")
                if len(self.headers.get_all("Content-Type", [])) != 1 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Send an application/json request.")
                charset = self.headers.get_content_charset("utf-8")
                if charset.replace("-", "").lower() != "utf8":
                    raise ValueError("JSON requests must use UTF-8.")
                body = self.read_body(length)
                payload = strict_json(body)
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object.")
                if set(payload) - {"text", "context", "include_jev"}:
                    raise ValueError("Only text, context and include_jev fields are supported.")
                prediction = compare_request(
                    artifacts,
                    payload.get("text"),
                    jev,
                    payload.get("include_jev", False),
                    payload.get("context", ""),
                )
                self.respond(200, prediction)
            except TimeoutError:
                self.respond(400, {"error": "Request body timed out before it was complete."})
            except ValueError as error:
                self.respond(400, {"error": str(error)})
            except ConnectionError:
                self.close_connection = True  # The browser disconnected while uploading.

        def log_message(self, format, *args):
            # Never log bodies, submitted excerpts, context, or provider secrets.
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts/method_models.joblib")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    try:
        # Only load trusted artifacts generated by this repository's train.py.
        artifacts = validate_artifacts(joblib.load(args.models))
        metrics_path = ROOT / "reports/metrics.json"
        metrics = strict_json(metrics_path.read_bytes()) if metrics_path.exists() else {}
        jev = JevClient()
        # Daemon request threads keep the demo responsive during slow uploads or
        # incomplete headers, without waiting for those clients at shutdown.
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(artifacts, metrics, jev))
    except (ValueError, OSError) as error:
        parser.error(f"{error} Run python prepare_data.py and python train.py first.")
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
