"""Serve the trained classifier on localhost for a course demonstration."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import joblib
from classifier import ARTIFACT_TASK
from jev import JevClient
from method_comparison import compare_request

ROOT = Path(__file__).resolve().parent


def make_handler(artifact, metrics, jev=None, slot_artifact=None):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.respond(200, (ROOT / "demo.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/results":
                self.respond(200, metrics)
            elif self.path == "/api/comparison-config":
                self.respond(200, jev.public_status() if jev else {"configured": False, "remaining_calls": 0})
            elif self.path == "/api/method-results":
                path = ROOT / "reports/method_comparison.json"
                self.respond(200, json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})
            else:
                self.respond(404, {"error": "Page not found."})

        def do_POST(self):
            if self.path != "/api/compare":
                return self.respond(404, {"error": "Page not found."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 25000:
                    raise ValueError("Request is too large or empty.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object.")
                prediction = compare_request(artifact, payload.get("text"), jev, payload.get("include_jev", False))
                if slot_artifact is not None:
                    from slots import display_spans, parser_tags, predict_tags
                    text = payload["text"]
                    tokens = text.split()
                    parser_spans = display_spans(text, parser_tags(tokens))
                    model_spans = display_spans(text, predict_tags(slot_artifact, [tokens])[0])
                    prediction["slots"] = {
                        "parser": {"model": "regex: date, time, timeofday only", "spans": parser_spans},
                        "token_model": {"model": slot_artifact["model_name"], "spans": model_spans},
                        "jev": {"message": "Jev slots are not implemented."},
                    }
                self.respond(200, prediction)
            except (ValueError, UnicodeDecodeError) as error:
                self.respond(400, {"error": str(error)})

        def log_message(self, format, *args):
            # Do not log the submitted text.
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/lr_model.joblib")
    parser.add_argument("--no-slots", action="store_true")
    parser.add_argument("--slot-model", type=Path, default=ROOT / "artifacts/slot_model.joblib")
    args = parser.parse_args()
    # Load only artifacts made locally by train.py. Joblib files can execute code.
    if not args.model.exists():
        parser.error("Train first: python train.py")
    artifact = joblib.load(args.model)
    if artifact.get("task") != ARTIFACT_TASK:
        parser.error("Train a MASSIVE intent model first: python train.py")
    metrics_path = ROOT / "reports/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    try:
        jev = JevClient()
    except ValueError as error:
        parser.error(str(error))
    # Slots are optional. Run with --no-slots if the slot extension was removed.
    slot_artifact = None
    if not args.no_slots and args.slot_model.exists():
        from slots import predict_tags
        slot_artifact = joblib.load(args.slot_model)
        try:
            predict_tags(slot_artifact, [])  # validates the artifact
        except ValueError as error:
            parser.error(str(error))
        if slot_artifact["dataset_sha256"] != artifact["dataset_sha256"]:
            parser.error("Intent and slot artifacts use different data.")
    server = HTTPServer(("127.0.0.1", args.port), make_handler(artifact, metrics, jev, slot_artifact))
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
