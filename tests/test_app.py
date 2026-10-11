"""Tests for app.py: origin and body checks, the Jev budget count, the request log, compare results, /api/results
built from a temporary folder, and the demo page. No model, no network, no server socket."""

import json
import re
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import app
import extract
from jev import JevClient

SERVER_ORIGIN = "http://127.0.0.1:8000"
PAGE = (Path(__file__).resolve().parent.parent / "demo.html").read_text(encoding="utf-8")


def body(**fields):
    return json.dumps(fields).encode("utf-8")


class FakeJev:
    """Counts classify calls and returns a fixed answer."""

    def __init__(self):
        self.calls = []

    def classify(self, text):
        self.calls.append(text)
        return {"status": "ok", "event_type": "sports", "tickets": "unknown",
                "confidence": {"event_type": 0.9, "tickets": 0.8}}


def fake_tagger(wanted):
    """A tagger that marks every token equal to a key of wanted with its slot, and nothing else."""
    return lambda tokens: [f"B-{wanted[t]}" if t in wanted else "O" for t in tokens]


MODELS = [
    extract.LoadedModel("dictionary", "Dictionary", "baseline", fake_tagger({"lakers": "event_name", "tmrw": "date"})),
    extract.LoadedModel("bertweet", "BERTweet", "transformer",
                     fake_tagger({"lakers": "event_name", "tmrw": "date", "boston": "city"})),
]
NOTES = ["BERT skipped: artifacts/bert not found"]
TEXT = "wanna catch the boston lakers game tmrw"


class HostTest(unittest.TestCase):
    def test_own_loopback_names_are_allowed(self):
        self.assertTrue(app.check_host("127.0.0.1:8765", 8765))
        self.assertTrue(app.check_host("localhost:8765", 8765))

    def test_other_hosts_and_ports_are_refused(self):
        for host in ("evil.example:8765", "127.0.0.1:8000", "localhost", "127.0.0.1", None, ""):
            self.assertFalse(app.check_host(host, 8765), host)


class OriginTest(unittest.TestCase):
    def test_missing_origin_is_allowed(self):
        self.assertTrue(app.check_origin(None, SERVER_ORIGIN))

    def test_own_origin_is_allowed(self):
        self.assertTrue(app.check_origin(SERVER_ORIGIN, SERVER_ORIGIN))

    def test_other_origins_are_refused(self):
        for origin in ("http://evil.example", "http://localhost:8000", "http://127.0.0.1:8001", "null", ""):
            self.assertFalse(app.check_origin(origin, SERVER_ORIGIN), origin)


class ParseBodyTest(unittest.TestCase):
    def assertRejected(self, raw, status):
        with self.assertRaises(app.RequestError) as context:
            app.parse_compare_body(raw)
        self.assertEqual(context.exception.status, status)

    def test_text_with_default_jev_flag(self):
        self.assertEqual(app.parse_compare_body(body(text="lakers game")), ("lakers game", False))

    def test_text_and_jev_flag(self):
        self.assertEqual(app.parse_compare_body(body(text="musicals", use_jev=True)), ("musicals", True))

    def test_bad_json_is_400(self):
        self.assertRejected(b"{not json", 400)

    def test_non_utf8_is_400(self):
        self.assertRejected(b'{"text": "\xff"}', 400)

    def test_non_object_is_400(self):
        self.assertRejected(b'["text"]', 400)

    def test_missing_or_non_string_text_is_400(self):
        self.assertRejected(body(use_jev=False), 400)
        self.assertRejected(body(text=5), 400)

    def test_non_boolean_use_jev_is_400(self):
        self.assertRejected(body(text="hi", use_jev="yes"), 400)

    def test_text_of_1000_characters_is_accepted(self):
        text, _ = app.parse_compare_body(body(text="a" * 1000))
        self.assertEqual(len(text), 1000)

    def test_text_over_1000_characters_is_413(self):
        self.assertRejected(body(text="a" * 1001), 413)

    def test_body_over_limit_is_413(self):
        self.assertRejected(b" " * (app.MAX_BODY_BYTES + 1), 413)


class JevBudgetTest(unittest.TestCase):
    def test_remaining_counts_free_receipt_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            jev = JevClient({"TYPESAFE_API_KEY": "test-key", "JEV_MAX_CALLS": "3"}, ledger_dir=directory)
            self.assertEqual(app.jev_remaining(jev), 3)
            (Path(directory) / "002.json").write_text("{}", encoding="utf-8")
            self.assertEqual(app.jev_remaining(jev), 2)


class RequestLogTest(unittest.TestCase):
    def test_append_creates_folders_and_writes_one_line_per_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data" / "user_requests" / "requests.jsonl"
            app.append_log(path, {"text": "concert next week?", "follow_ups": ["ask_city"]})
            app.append_log(path, {"text": "Zürich café", "follow_ups": []})
            raw = path.read_bytes()
            self.assertNotIn(b"\r", raw)
            lines = raw.decode("utf-8").split("\n")
            self.assertEqual(lines[-1], "")
            self.assertEqual(json.loads(lines[0]), {"text": "concert next week?", "follow_ups": ["ask_city"]})
            self.assertIn("Zürich café", lines[1])

    def test_append_adds_to_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text('{"old": true}\n', encoding="utf-8")
            app.append_log(path, {"new": True})
            self.assertEqual(path.read_text(encoding="utf-8").splitlines(), ['{"old": true}', '{"new": true}'])


class CompareTest(unittest.TestCase):
    def run_compare(self, use_jev, jev=None):
        return app.compare_text(TEXT, use_jev, jev, MODELS, NOTES, threading.Lock())

    def test_result_has_every_model_and_the_notes(self):
        result = self.run_compare(False)
        self.assertEqual([row["name"] for row in result["models"]], ["Dictionary", "BERTweet"])
        self.assertEqual(result["notes"], NOTES)
        self.assertEqual(result["models"][1]["slots"], {"city": ["boston"], "date": ["tmrw"], "event_name": ["lakers"]})

    def test_jev_is_skipped_without_the_flag(self):
        jev = FakeJev()
        self.assertEqual(self.run_compare(False, jev)["jev"], {"status": "skipped"})
        self.assertEqual(jev.calls, [])

    def test_jev_is_called_once_for_all_rows(self):
        jev = FakeJev()
        result = self.run_compare(True, jev)
        self.assertEqual(jev.calls, [TEXT])
        self.assertEqual(result["jev"]["event_type"], "sports")

    def test_blank_text_never_reaches_jev_and_gives_no_spans(self):
        jev = FakeJev()
        result = app.compare_text("   ", True, jev, MODELS, NOTES, threading.Lock())
        self.assertEqual(jev.calls, [])
        self.assertEqual(result["jev"], {"status": "empty"})
        self.assertEqual(result["models"][0]["spans"], [])

    def test_log_record_has_bertweet_slots_and_every_model(self):
        now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
        record = app.build_log_record(self.run_compare(False), now)
        self.assertEqual(record["time_utc"], "2026-10-10T12:00:00Z")
        self.assertEqual(record["endpoint"], "compare")
        self.assertEqual(record["slots"]["city"], ["boston"])
        self.assertEqual(record["models"]["dictionary"], {"city": [], "date": ["tmrw"], "event_name": ["lakers"]})
        self.assertEqual(record["jev"], {"status": "skipped"})

    def test_log_record_keeps_only_the_jev_summary(self):
        now = datetime(2026, 10, 10, 9, 30, 5, tzinfo=timezone(timedelta(hours=3)))
        result = self.run_compare(False)
        result["jev"] = {"status": "ok", "event_type": "sports", "tickets": "two", "extra": "dropped",
                         "confidence": {}}
        record = app.build_log_record(result, now)
        self.assertEqual(record["time_utc"], "2026-10-10T06:30:05Z")
        self.assertEqual(record["jev"], {"status": "ok", "event_type": "sports", "tickets": "two"})
        result["jev"] = {"status": "error", "message": "Jev failed"}
        self.assertEqual(app.build_log_record(result, now)["jev"], {"status": "error"})


CLASSIC_RESULTS = {"seed": 42, "results": [
    {"training_set": "D: SGD+MASSIVE+typos", "model": "MultinomialNB", "setting": "alpha=0.3", "dev_f1": 0.7,
     "scores": {"SGD test": {"f1": 0.8}, "MASSIVE test": {"f1": 0.6},
                "messy": {"f1": 0.5, "slots": {"event_name": {"f1": 0.2}}}}},
], "learning_curve": {"model": "LinearSVC", "points": [{"fraction": 0.1, "sentences": 100, "test_f1": 0.8}]}}
BERT = {"model": "bert-base-uncased", "test": {"SGD test": {"f1": 0.97}, "MASSIVE test": {"f1": 0.9},
                                              "messy": {"f1": 0.85, "slots": {"event_name": {"f1": 0.7}}}}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ResultsTest(unittest.TestCase):
    def test_missing_files_are_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(app.build_results(Path(tmp), logged=None), {})

    def test_reports_and_data_are_summarised(self):
        messy_row = {"id": "messy:02", "text": "shows in nyc", "tokens": ["shows", "in", "nyc"],
                     "tags": ["O", "O", "B-city"]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "results" / "classic_results.json", CLASSIC_RESULTS)
            write_json(root / "results" / "bert_results.json", BERT)
            (root / "data" / "messy").mkdir(parents=True)
            (root / "data" / "messy" / "messy_test.jsonl").write_text(
                json.dumps(messy_row) + "\n" + json.dumps({**messy_row, "id": "messy:03"}) + "\n", encoding="utf-8")
            out = app.build_results(root, logged=2)

        self.assertEqual([(s["name"], s["train"], s["test"]) for s in out["data"]], [("Messy test", 0, 2)])
        self.assertEqual(out["data"][0]["example"], {"text": "shows in nyc",
                                                     "spans": [{"slot": "city", "start": 9, "end": 12}]})
        self.assertEqual(out["user_requests"], 2)
        self.assertEqual(out["classic"][0]["name"], "Naive Bayes")
        self.assertEqual(out["classic"][0]["messy_slots"], {"event_name": 0.2})
        self.assertEqual(out["transformers"][0]["name"], "BERT")
        self.assertEqual(out["transformers"][0]["messy_f1"], 0.85)
        self.assertEqual(out["seed"], CLASSIC_RESULTS["seed"])

    def test_distillation_shows_the_alpha_with_the_best_dev_f1(self):
        def run(dev, messy, alpha=None):
            return {**BERT, "dev_f1_chosen": dev, "distillation": {"alpha": alpha} if alpha else None,
                    "test": {**BERT["test"], "messy": {"f1": messy, "slots": {}}}}
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            write_json(results / "bert_mini_8ep_E_results.json", run(0.95, 0.80))
            write_json(results / "bert_mini_8ep_E_kd0.3_results.json", run(0.96, 0.70, 0.3))
            write_json(results / "bert_mini_8ep_E_kd0.5_results.json", run(0.94, 0.90, 0.5))
            groups = app.build_results(Path(tmp), logged=None)["distillation"]

        self.assertEqual(len(groups), 1)
        gold, soft = groups[0]["rows"]
        self.assertEqual((gold["soft"], gold["messy_f1"]), (False, 0.80))
        self.assertEqual((soft["soft"], soft["messy_f1"], soft["dev_f1"]), (True, 0.70, 0.96))
        self.assertTrue(soft["name"].endswith("α 0.3"))

    def test_shrinking_groups_precisions_by_model(self):
        def run(model, precision, mb, messy):
            return {"model": model, "precision": precision, "parameters": 1, "megabytes": mb,
                    "f1": {"dev": 0.9, "messy": messy, "codex": 0.8}}
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / "results" / "quantized_results.json",
                       [run("bertweet_6L_E_kd0.7", "float32", 367.2, 0.95),
                        run("bertweet_6L_E_kd0.7", "int8", 93.2, 0.9), run("other_model", "float16", 1.0, 0.5)])
            groups = app.build_results(Path(tmp), logged=None)["shrinking"]

        self.assertEqual([g["heading"] for g in groups], ["BERTweet, 6 layers"])
        self.assertEqual([r["name"] for r in groups[0]["rows"]], ["32-bit, 367 MB", "8-bit, 93 MB"])
        self.assertEqual(groups[0]["rows"][1]["messy_f1"], 0.9)

    def test_jev_teacher_check_is_summarized(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / "results" / "jev_teacher_check.json",
                       {"by_group": {"all": {"sentences": 300, "jev": {"f1": 0.73}, "bertweet": {"f1": 0.96}}},
                        "ledger_usage": {"attempts": 310, "estimated_input_cost_usd": 0.04}})
            jev = app.build_results(Path(tmp), logged=None)["jev_teacher"]

        self.assertEqual(jev, {"sentences": 300, "jev_f1": 0.73, "bertweet_f1": 0.96, "calls": 310, "cost_usd": 0.04})

    def test_pick_collects_16_bit_seeds_per_model(self):
        def run(model, precision, messy, ms):
            return {"model": model, "precision": precision, "parameters": 1, "megabytes": 184.0,
                    "ms_per_request": ms, "f1": {"dev": 0.9, "messy": messy, "codex": 0.8}}
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / "results" / "quantized_results.json",
                       [run("bertweet_6L_E_kd0.7", "float16", 0.95, 3.0),
                        run("bertweet_6L_E_kd0.7_seed1", "float16", 0.91, 4.0),
                        run("bertweet_6L_E_kd0.7_seed1", "int8", 0.5, 9.0),
                        run("bertweet_6L_E_kd0.3", "float16", 0.1, 1.0)])
            pick = app.build_results(Path(tmp), logged=None)["pick"]

        self.assertEqual([p["name"] for p in pick], ["6 layers"])
        self.assertEqual(pick[0]["messy"], [0.95, 0.91])
        self.assertEqual(pick[0]["ms_per_request"], 3.5)

    def test_data_curve_averages_seeds_per_size(self):
        def run(messy, codex):
            return {"test": {"messy": {"f1": messy}, "codex_all": {"f1": codex}}}
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            write_json(results / "bertweet_syn0_results.json", run(0.8, 0.82))
            write_json(results / "bertweet_syn0_seed1_results.json", run(0.7, 0.84))
            write_json(results / "bertweet_syn250_results.json", run(0.9, 0.9))
            curve = app.build_results(Path(tmp), logged=None)["data_curve"]

        self.assertEqual([(p["rows"], p["runs"]) for p in curve], [(0, 2), (250, 1)])
        self.assertAlmostEqual(curve[0]["messy_mean"], 0.75)
        self.assertAlmostEqual(curve[0]["codex_mean"], 0.83)
        self.assertGreater(curve[0]["messy_sd"], 0)
        self.assertEqual(curve[1]["codex_sd"], 0.0)

    def test_request_count_is_left_out_when_logging_is_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / "results" / "classic_results.json", CLASSIC_RESULTS)
            self.assertNotIn("user_requests", app.build_results(Path(tmp), logged=None))


class PageTest(unittest.TestCase):
    def test_page_has_title_and_both_tabs(self):
        self.assertIn("<title>Event request extractor</title>", PAGE)
        self.assertIn('id="tab-try"', PAGE)
        self.assertIn('id="tab-results"', PAGE)

    def test_page_loads_no_external_resources(self):
        self.assertIsNone(re.search(r"(src|href)\s*=\s*[\"']?\s*https?:|url\(\s*['\"]?https?:|@import|<link",
                                    PAGE, re.IGNORECASE))

    def test_page_builds_the_dom_without_innerhtml(self):
        self.assertNotIn("innerHTML", PAGE)
        self.assertIn("createElementNS", PAGE)

    def test_page_shows_request_log_note_and_every_follow_up(self):
        self.assertIn("Requests are saved locally to data/user_requests/ for testing.", PAGE)
        for text in ("Ask for the city", "Ask for the date", "Ask what kind of event", "Ask how many people",
                     "None, ready to search"):
            self.assertIn(text, PAGE)

    def test_page_compares_models_on_the_same_data(self):
        self.assertFalse("Best messy F1" in PAGE)
        self.assertIn("All trained on set D.", PAGE)

    def test_page_only_calls_the_three_api_routes(self):
        self.assertEqual(sorted(set(re.findall(r'fetch\("(/api/[a-z]+)"', PAGE))),
                         ["/api/compare", "/api/results", "/api/status"])


if __name__ == "__main__":
    unittest.main()
