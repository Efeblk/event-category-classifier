"""Serve the trained classifier on localhost for a course demonstration."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent


def predict_event(artifact, text):
    if not isinstance(text, str) or not text.strip() or len(text) > 5000:
        raise ValueError("Enter an event title or description with 1 to 5,000 characters.")
    model = artifact["model"]
    result = {"label": str(model.predict([text])[0]), "model": artifact["model_name"]}
    if hasattr(model, "predict_proba"):
        result["probabilities"] = {
            str(label): round(float(score), 4)
            for label, score in zip(model.classes_, model.predict_proba([text])[0])
        }
    return result


def make_handler(artifact, metrics):
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
            else:
                self.respond(404, {"error": "Page not found."})

        def do_POST(self):
            if self.path != "/api/predict":
                return self.respond(404, {"error": "Page not found."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 25000:
                    raise ValueError("Request is too large or empty.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object.")
                self.respond(200, predict_event(artifact, payload.get("text")))
            except (ValueError, UnicodeDecodeError) as error:
                self.respond(400, {"error": str(error)})

        def log_message(self, format, *args):
            # Do not log the submitted event description.
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/model.joblib")
    args = parser.parse_args()
    # Load only artifacts made locally by train.py. Joblib files can execute code.
    if not args.model.exists():
        parser.error("Train the models first: python train.py")
    artifact = joblib.load(args.model)
    metrics_path = ROOT / "reports/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    server = HTTPServer(("127.0.0.1", args.port), make_handler(artifact, metrics))
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
