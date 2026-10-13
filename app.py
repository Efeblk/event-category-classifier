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
import re
import statistics
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import extract
from jev import JevClient
from slots import read_jsonl

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "demo.html"
DEFAULT_LOG_PATH = ROOT / "data" / "user_requests" / "requests.jsonl"
LOG_LOCK = threading.Lock()
MAX_TEXT_CHARACTERS = 1000
MAX_BODY_BYTES = 16_384
MODEL_NAMES = {"dictionary baseline": "Parser", "MultinomialNB": "Naive Bayes",
               "LogisticRegression": "Logistic Regression", "LinearSVC": "Linear SVM"}
# Distillation chart: (heading, short name, [(results key, row name)]). A key ending in _kd stands for all
# <key><alpha> runs; the one with the best dev F1 is shown, so alpha is chosen on dev like every other setting.
DISTILLATION = (
    ("Same size: BERTweet learns from RoBERTa-large", "BERTweet", (
        ("roberta_large_E", "RoBERTa-large (teacher, 355M)"),
        ("bertweet_E", "BERTweet, gold labels (135M)"),
        ("bertweet_E_kd", "BERTweet, + soft labels"))),
    ("12× smaller: BERT-mini learns from BERTweet", "BERT-mini", (
        ("bertweet_E", "BERTweet (teacher, 135M)"),
        ("bert_mini_8ep_E", "BERT-mini, gold labels (11M)"),
        ("bert_mini_8ep_E_kd", "BERT-mini, + soft labels"))),
    ("12× smaller, plus 9k messy copies (set F): BERT-mini learns from BERTweet", "BERT-mini + copies", (
        ("bertweet_E", "BERTweet (teacher, 135M)"),
        ("bert_mini_8ep_F", "BERT-mini, gold labels"),
        ("bert_mini_8ep_F_kd", "BERT-mini, + soft labels"))),
)


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
    """One request-log entry: the main model's slots, every model's slots and the Jev summary. now is
    timezone-aware."""
    main = next((row for row in result["models"] if row["key"] == extract.MAIN_MODEL), {})
    return {"time_utc": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "text": result["text"], "endpoint": "compare",
            "slots": main.get("slots"), "spans": main.get("spans"),
            "follow_ups": main.get("follow_ups", []),
            "models": {row["key"]: row["slots"] for row in result["models"]},
            "jev": jev_summary(result["jev"])}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def count_lines(path):
    return sum(1 for line in path.read_text(encoding="utf-8").split("\n") if line.strip())


# Data section: (name, file under data/, test file or None, id of the example row, what the source is, where its
# tags come from). Typo copies have no file of their own: they are made from SGD and MASSIVE by
# train_classic.typo_copy.
DATA_SOURCES = (
    ("SGD Events (Google)", "clean/train.jsonl", "clean/test.jsonl", "61_00000:2",
     "Clean requests to a virtual assistant",
     "Google's annotators marked each slot by its character positions"),
    ("MASSIVE (Amazon)", "clean/massive_train.jsonl", "clean/massive_test.jsonl", "massive:3802",
     "Short assistant commands; only dates and places, mostly not about events",
     "Amazon's annotators marked slots in brackets, like [date : tomorrow]"),
    ("Typo copies", None, None, "61_00000:2:typo", "A copy of every SGD and MASSIVE sentence with random letter typos",
     "Copied from the original sentence: typos change letters, not words"),
    ("AI-written requests (Claude Haiku)", "clean/synthetic_train.jsonl", None, "synthetic:part1:115",
     "Casual, messy requests: slang, lowercase, emojis",
     "Haiku wrote each request with its slots; the script checks every slot is in the text and drops rows "
     "naming a test event"),
    ("Messy test", None, "messy/messy_test.jsonl", "messy:02", "Drafted by AI, checked by hand; never trained on",
     "Drafted by AI, then checked and corrected by hand"),
    ("Codex test", None, "codex/codex_test.jsonl", "codex:21", "Written by a different AI, checked by hand",
     "Written by OpenAI Codex, checked by hand (9 of 60 labels fixed)"),
)
# Shrinking chart: names for the models in results/quantized_results.json, in the order they are shown.
SHRINK_NAMES = {"bertweet_E": "BERTweet, all 12 layers", "bertweet_6L_E_kd0.7": "BERTweet, 6 layers",
                "bertweet_4L_bottom_E_kd0.3": "BERTweet, bottom 4 layers"}
PRECISION_NAMES = {"float32": "32-bit", "float16": "16-bit", "int8": "8-bit"}
# Section 5.1: the two 16-bit versions compared over every seed in quantized_results.json.
PICK_NAMES = {"bertweet_E": "12 layers", "bertweet_6L_E_kd0.7": "6 layers"}


def f1_summary(scores):
    """SGD, MASSIVE, messy and Codex F1 plus the messy per-slot F1 from one set of span scores."""
    return {"sgd_f1": scores["SGD test"]["f1"], "massive_f1": scores["MASSIVE test"]["f1"],
            "messy_f1": scores["messy"]["f1"], "codex_f1": scores.get("codex", {}).get("f1"),
            "messy_slots": {slot: info["f1"] for slot, info in scores["messy"]["slots"].items()}}


def build_transformers(results_dir):
    """F1 summary of every transformer in extract.TRANSFORMER_MODELS whose results exist."""
    rows = []
    for key, name in extract.TRANSFORMER_MODELS:
        found = read_json(results_dir / f"{key}_results.json")
        if found is not None:
            rows.append({"key": key, "name": name, "model": found["model"], **f1_summary(found["test"])})
    return rows


def distillation_run(results_dir, key):
    """The results of one run; for a key ending in _kd, the alpha run with the best dev F1. None if missing."""
    if not key.endswith("_kd"):
        return read_json(results_dir / f"{key}_results.json")
    runs = [read_json(path) for path in sorted(results_dir.glob(f"{key}*_results.json"))]
    return max(runs, key=lambda run: run["dev_f1_chosen"], default=None)


def build_distillation(results_dir):
    """Groups of rows for the distillation chart; rows whose results are missing are left out."""
    groups = []
    for heading, short, rows in DISTILLATION:
        found = []
        for key, name in rows:
            run = distillation_run(results_dir, key)
            if run is not None:
                alpha = (run.get("distillation") or {}).get("alpha")
                found.append({"name": name + (f", α {alpha}" if alpha is not None else ""), "soft": alpha is not None,
                              "dev_f1": run["dev_f1_chosen"], **f1_summary(run["test"])})
        if found:
            groups.append({"heading": heading, "short": short, "rows": found})
    return groups


def build_shrinking(results_dir):
    """Groups of rows (one per stored precision) for the shrinking chart, from quantized_results.json."""
    runs = read_json(results_dir / "quantized_results.json") or []
    groups = []
    for key, heading in SHRINK_NAMES.items():
        rows = [{"name": f"{PRECISION_NAMES[r['precision']]}, {r['megabytes']:.0f} MB", "megabytes": r["megabytes"],
                 "dev_f1": r["f1"]["dev"], "messy_f1": r["f1"]["messy"], "codex_f1": r["f1"]["codex"]}
                for r in runs if r["model"] == key]
        if rows:
            groups.append({"heading": heading, "rows": rows})
    return groups


def build_pick(results_dir):
    """The 16-bit 12- and 6-layer BERTweet over seeds (key or key_seed<n>): size, median ms per request and each
    seed's messy and Codex F1. Models with no 16-bit rows are left out."""
    runs = read_json(results_dir / "quantized_results.json") or []
    out = []
    for key, name in PICK_NAMES.items():
        rows = [r for r in runs if r["precision"] == "float16" and re.fullmatch(re.escape(key) + r"(_seed\d+)?",
                                                                                r["model"])]
        if rows:
            out.append({"name": name, "megabytes": rows[0]["megabytes"],
                        "ms_per_request": statistics.median(r["ms_per_request"] for r in rows),
                        "messy": [r["f1"]["messy"] for r in rows], "codex": [r["f1"]["codex"] for r in rows]})
    return out


def build_data_curve(results_dir):
    """Points of the data curve (train_bert.py --synthetic N): per N, the mean and standard deviation over seeds of
    messy and Codex (all 260) F1, from bertweet_syn<N>[_seed<n>]_results.json."""
    by_n = {}
    for path in results_dir.glob("bertweet_syn*_results.json"):
        n = int(re.match(r"bertweet_syn(\d+)", path.name).group(1))
        by_n.setdefault(n, []).append(read_json(path)["test"])
    points = []
    for n in sorted(by_n):
        point = {"rows": n, "runs": len(by_n[n])}
        for name, key in (("messy", "messy"), ("codex_all", "codex")):
            values = [test[name]["f1"] for test in by_n[n] if name in test]
            point[key + "_mean"] = statistics.mean(values) if values else None
            point[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        points.append(point)
    return points


def example_view(row):
    """Text and character spans of one data row, for highlighting. Rows whose tokens no longer match their text
    (typo copies) are shown as the tokens joined by spaces."""
    text = row["text"] if extract.TOKEN.findall(row["text"]) == row["tokens"] else " ".join(row["tokens"])
    return {"text": text, "spans": extract.extract_spans(text, lambda tokens: row["tags"])}


def build_data(data):
    """Sources of the Data section with sentence counts and one example each; missing files are left out."""
    sources = []
    for name, train_file, test_file, example_id, note, labels in DATA_SOURCES:
        train = read_jsonl_rows(data / train_file) if train_file else None
        test = read_jsonl_rows(data / test_file) if test_file else None
        if name == "Typo copies":
            sgd = read_jsonl_rows(data / "clean" / "train.jsonl")
            massive = read_jsonl_rows(data / "clean" / "massive_train.jsonl")
            if sgd is None or massive is None:
                continue
            from train_classic import typo_copy
            train = typo_copy(sgd)
            train_count = len(sgd) + len(massive)
        elif train is None and test is None:
            continue
        else:
            train_count = len(train) if train is not None else 0
        rows = train if train is not None else test
        example = next((r for r in rows if r["id"] == example_id), None)
        sources.append({"name": name, "note": note, "labels": labels, "train": train_count,
                        "test": len(test) if test is not None else 0,
                        "example": example_view(example) if example else None})
    return sources


def read_jsonl_rows(path):
    """The rows of a JSON-lines file, or None when it does not exist."""
    return read_jsonl(path) if path.exists() else None


def build_results(root=ROOT, logged=None):
    """Everything the Results tab shows, from the files that exist. Missing files are left out of the result.

    logged is the number of requests in the request log, or None when logging is off.
    """
    results_dir, data = Path(root) / "results", Path(root) / "data"
    out = {}
    synthetic = read_json(data / "clean" / "synthetic_stats.json")
    if synthetic is not None:
        out["synthetic_stats"] = {key: synthetic[key] for key in ("rows_in", "dropped_test_event_names", "rows_out")}
    if logged is not None:
        out["user_requests"] = logged

    classic = read_json(results_dir / "classic_results.json")
    if classic is not None:
        out["seed"] = classic["seed"]
        out["classic"] = [{"training_set": r["training_set"], "model": r["model"],
                           "name": MODEL_NAMES.get(r["model"], r["model"]), "setting": r["setting"],
                           "dev_f1": r["dev_f1"], **f1_summary(r["scores"])} for r in classic["results"]]

    jev = read_json(results_dir / "jev_teacher_check.json")
    if jev is not None:
        scores = jev["by_group"]["all"]
        out["jev_teacher"] = {"sentences": scores["sentences"], "jev_f1": scores["jev"]["f1"],
                              "bertweet_f1": scores["bertweet"]["f1"], "calls": jev["ledger_usage"]["attempts"],
                              "cost_usd": jev["ledger_usage"]["estimated_input_cost_usd"]}
    lists = {"data": build_data(data), "transformers": build_transformers(results_dir),
             "distillation": build_distillation(results_dir), "shrinking": build_shrinking(results_dir),
             "pick": build_pick(results_dir), "data_curve": build_data_curve(results_dir)}
    out.update({key: value for key, value in lists.items() if value})
    return out


def compare_text(text, use_jev, jev, models, notes, lock, vocabulary=frozenset()):
    """Every model on text, plus the words no model saw in training. Jev is asked at most once here and its answer
    is shared by all rows."""
    jev_result = extract.jev_check(text, jev if use_jev else None)
    with lock:
        rows = extract.compare(text, models, jev_result)
    return {"text": text, "jev": jev_result, "models": rows, "notes": notes,
            "new_words": extract.new_words(text, vocabulary) if vocabulary else []}


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port, models, notes, jev, log_path, vocabulary=frozenset()):
        super().__init__(("127.0.0.1", port), Handler)
        self.models, self.notes, self.jev, self.log_path = models, notes, jev, log_path
        self.vocabulary = vocabulary
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
                              self.server.gpu_lock, self.server.vocabulary)
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
    server = DemoServer(args.port, models, notes, jev, args.log_path if args.log else None,
                        extract.training_vocabulary())
    print(f"ready on {server.origin}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
