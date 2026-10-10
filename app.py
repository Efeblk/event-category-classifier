"""Local demo for the event request extractor, served from demo.html.

The Try it tab sends one request to POST /api/compare, which runs every slot tagger and shows each model's slot
highlights side by side, with an optional Jev check. The Results tab draws charts from GET /api/results, which
summarises the JSON files in results/ and the stats in data/ (missing files are left out).

Usage: python app.py [--port 8000] [--no-jev] [--log] [--log-path FILE]
The server listens on 127.0.0.1 only. The models load once at start and run under one GPU lock. Requests must come
from no origin or from this server's own origin, and GET and POST requests must name this server in the Host header
(127.0.0.1:PORT or localhost:PORT). Request logging is off unless --log is given. With --log, each compare call
appends one JSON line to data/user_requests/requests.jsonl (or --log-path).
Standard library only; torch loads through extract.load_models.
"""

import argparse
import json
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import extract
from jev import JevClient

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "demo.html"
DEFAULT_LOG_PATH = ROOT / "data" / "user_requests" / "requests.jsonl"
LOG_LOCK = threading.Lock()
MAX_TEXT_CHARACTERS = 1000
MAX_BODY_BYTES = 16_384
MODEL_NAMES = {"dictionary baseline": "Dictionary", "MultinomialNB": "Naive Bayes",
               "LogisticRegression": "Logistic Regression", "LinearSVC": "Linear SVM"}


class RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def check_origin(origin, server_origin):
    """Accept requests without an Origin header or with this server's own origin."""
    return origin is None or origin == server_origin


def check_host(host, port):
    """Accept only a Host header that names this server, which blocks DNS rebinding from other sites."""
    return host in {f"127.0.0.1:{port}", f"localhost:{port}"}


def parse_compare_body(raw):
    """Return (text, use_jev) from a JSON request body, or raise RequestError."""
    if len(raw) > MAX_BODY_BYTES:
        raise RequestError(413, "Request body is too large.")
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError as error:
        raise RequestError(400, "Body must be UTF-8 JSON.") from error
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise RequestError(400, 'Body must be an object with a "text" string.')
    use_jev = data.get("use_jev", False)
    if not isinstance(use_jev, bool):
        raise RequestError(400, '"use_jev" must be true or false.')
    if len(data["text"]) > MAX_TEXT_CHARACTERS:
        raise RequestError(413, f"Text is longer than {MAX_TEXT_CHARACTERS} characters.")
    return data["text"], use_jev


def jev_remaining(jev):
    """Calls left in the shared budget: receipt numbers 001 to JEV_MAX_CALLS that are still free."""
    return sum(not (jev.ledger_dir / f"{index:03d}.json").exists() for index in range(1, jev.max_calls + 1))


def append_log(path, record):
    """Append one JSON line to the request log; a lock keeps concurrent requests from interleaving."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with LOG_LOCK:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as file:
            file.write(line)


def jev_summary(jev_result):
    """Jev fields worth keeping in the log: the answers when Jev was ok, otherwise just the status."""
    if jev_result.get("status") == "ok":
        return {"status": "ok", "event_type": jev_result["event_type"], "tickets": jev_result["tickets"]}
    return {"status": jev_result.get("status")}


def build_log_record(result, now):
    """One request-log entry: the BERTweet slots, every model's slots and the Jev summary. now is timezone-aware."""
    bertweet = next((row for row in result["models"] if row["key"] == "bertweet"), {})
    return {"time_utc": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "text": result["text"], "endpoint": "compare",
            "slots": bertweet.get("slots"), "spans": bertweet.get("spans"),
            "follow_ups": bertweet.get("follow_ups", []),
            "models": {row["key"]: row["slots"] for row in result["models"]},
            "jev": jev_summary(result["jev"])}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def count_lines(path):
    return sum(1 for line in path.read_text(encoding="utf-8").split("\n") if line.strip())


def f1_summary(scores):
    """SGD, MASSIVE and messy F1 plus the messy per-slot F1 from one set of span scores."""
    return {"sgd_f1": scores["SGD test"]["f1"], "massive_f1": scores["MASSIVE test"]["f1"],
            "messy_f1": scores["messy"]["f1"],
            "messy_slots": {slot: info["f1"] for slot, info in scores["messy"]["slots"].items()}}


def build_results(root=ROOT, logged=None):
    """Everything the Results tab shows, from the files that exist. Missing files are left out of the result.

    logged is the number of requests in the request log, or None when logging is off.
    """
    results_dir, data = Path(root) / "results", Path(root) / "data"
    out = {}
    stats = {key: read_json(data / "clean" / name)
             for key, name in (("sgd", "stats.json"), ("massive", "massive_stats.json"))}
    stats = {key: value for key, value in stats.items() if value is not None}
    if stats:
        out["stats"] = stats
    messy = data / "messy" / "messy_test.jsonl"
    if messy.exists():
        out["messy_test_rows"] = count_lines(messy)
    if logged is not None:
        out["user_requests"] = logged

    classic = read_json(results_dir / "classic_results.json")
    if classic is not None:
        out["seed"] = classic["seed"]
        out["classic"] = [{"training_set": r["training_set"], "model": r["model"],
                           "name": MODEL_NAMES.get(r["model"], r["model"]), "setting": r["setting"],
                           "dev_f1": r["dev_f1"], **f1_summary(r["scores"])} for r in classic["results"]]
        out["learning_curve"] = classic["learning_curve"]["points"]

    transformers = []
    for key, name in extract.TRANSFORMER_MODELS:
        found = read_json(results_dir / f"{key}_results.json")
        if found is not None:
            transformers.append({"key": key, "name": name, "model": found["model"], **f1_summary(found["test"])})
    if transformers:
        out["transformers"] = transformers
    return out


def compare_text(text, use_jev, jev, models, notes, lock):
    """Every model on text. Jev is asked at most once here and its answer is shared by all rows."""
    jev_result = extract.jev_check(text, jev if use_jev else None)
    with lock:
        rows = extract.compare(text, models, jev_result)
    return {"text": text, "jev": jev_result, "models": rows, "notes": notes}


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port, models, notes, jev, log_path):
        super().__init__(("127.0.0.1", port), Handler)
        self.models, self.notes, self.jev, self.log_path = models, notes, jev, log_path
        self.gpu_lock = threading.Lock()
        # Counted once at start; each saved request adds one, so GET /api/results never rereads the log.
        self.logged = count_lines(Path(log_path)) if log_path is not None and Path(log_path).exists() else 0

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def status(self):
        configured = self.jev is not None and self.jev.configured
        return {"jev_configured": configured, "jev_off": self.jev is None,
                "jev_remaining": jev_remaining(self.jev) if configured else None,
                "request_log": self.log_path is not None}

    def log(self, record):
        if self.log_path is None:
            return
        try:
            append_log(self.log_path, record)
        except OSError as error:
            print(f"Request log not saved: {error}", file=sys.stderr, flush=True)
            return
        with LOG_LOCK:
            self.logged += 1


class Handler(BaseHTTPRequestHandler):
    server: DemoServer

    def host_allowed(self):
        return check_host(self.headers.get("Host"), self.server.server_address[1])

    def do_GET(self):
        if not self.host_allowed():
            return self._json(403, {"error": "Unexpected Host header."})
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif self.path == "/api/status":
            self._json(200, self.server.status())
        elif self.path == "/api/results":
            logged = self.server.logged if self.server.log_path is not None else None
            self._json(200, build_results(ROOT, logged))
        else:
            self._json(404, {"error": "Not found."})

    def do_POST(self):
        if not self.host_allowed():
            return self._json(403, {"error": "Unexpected Host header."})
        if not check_origin(self.headers.get("Origin"), self.server.origin):
            return self._json(403, {"error": "Cross-origin requests are not allowed."})
        if self.path != "/api/compare":
            return self._json(404, {"error": "Not found."})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            return self._json(400, {"error": "Invalid Content-Length."})
        try:
            text, use_jev = parse_compare_body(self.rfile.read(min(length, MAX_BODY_BYTES + 1)))
        except RequestError as error:
            return self._json(error.status, {"error": error.message})
        result = compare_text(text, use_jev, self.server.jev, self.server.models, self.server.notes,
                              self.server.gpu_lock)
        self.server.log(build_log_record(result, datetime.now(timezone.utc)))
        self._json(200, result)

    def _json(self, status, payload):
        self._send(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000, help="port on 127.0.0.1")
    parser.add_argument("--no-jev", action="store_true", help="never call Jev")
    parser.add_argument("--log", action="store_true", help="save each compare request to the request log")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH,
                        help="request log file (JSON lines), used with --log")
    args = parser.parse_args(argv)

    models, notes = extract.load_models()
    for note in notes:
        print(note, flush=True)
    jev = None if args.no_jev else JevClient()
    server = DemoServer(args.port, models, notes, jev, args.log_path if args.log else None)
    print(f"ready on {server.origin}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
