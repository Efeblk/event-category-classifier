"""Tests for extract.py with fake taggers: no model is loaded, no torch, no network."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import extract
from slots import DictionaryTagger

TAGS = ["B-event_name", "I-event_name", "O", "B-city", "I-city", "I-city", "B-date", "I-date", "O", "O"]
TEXT = "taylor swift in chi-town this wknd??"


class SpanTest(unittest.TestCase):
    def test_span_text_keeps_original_characters(self):
        found = extract.group_slots(TEXT, extract.extract_spans(TEXT, lambda tokens: TAGS))
        self.assertEqual(found, {"city": ["chi-town"], "date": ["this wknd"], "event_name": ["taylor swift"]})

    def test_offsets_point_into_the_original_text(self):
        spans = extract.extract_spans(TEXT, lambda tokens: TAGS)
        self.assertEqual(spans, [
            {"slot": "event_name", "start": 0, "end": 12},
            {"slot": "city", "start": 16, "end": 24},
            {"slot": "date", "start": 25, "end": 34},
        ])

    def test_original_spacing_is_preserved(self):
        tags = ["B-city", "I-city", "I-city"]
        found = extract.group_slots("chi  -  town", extract.extract_spans("chi  -  town", lambda t: tags))
        self.assertEqual(found["city"], ["chi  -  town"])

    def test_tagger_receives_word_and_punctuation_tokens(self):
        seen = []

        def tag(tokens):
            seen.append(tokens)
            return ["O"] * len(tokens)

        extract.extract_spans("chi-town, 2026!", tag)
        self.assertEqual(seen, [["chi", "-", "town", ",", "2026", "!"]])

    def test_orphan_inside_tag_is_repaired_to_a_span(self):
        found = extract.group_slots("New York", extract.extract_spans("New York", lambda t: ["I-city", "I-city"]))
        self.assertEqual(found["city"], ["New York"])

    def test_blank_text_does_not_call_the_tagger(self):
        calls = []
        self.assertEqual(extract.extract_spans("   ", lambda tokens: calls.append(tokens)), [])
        self.assertEqual(calls, [])


class JevCheckTest(unittest.TestCase):
    def test_skipped_without_a_client(self):
        self.assertEqual(extract.jev_check("lakers game"), {"status": "skipped"})

    def test_blank_text_is_not_sent_to_jev(self):
        class NoCallJev:
            def classify(self, text):
                raise AssertionError("Jev must not be called for blank text")

        self.assertEqual(extract.jev_check("  ", NoCallJev()), {"status": "empty"})


class FollowUpsTest(unittest.TestCase):
    OK_MUSIC_TWO = {"status": "ok", "event_type": "music", "tickets": "two"}
    OK_UNKNOWN_TWO = {"status": "ok", "event_type": "unknown", "tickets": "two"}
    OK_MUSIC_UNKNOWN_TICKETS = {"status": "ok", "event_type": "music", "tickets": "unknown"}
    FAILED_JEV = ({"status": "error", "message": "x"}, {"status": "skipped"}, {"status": "not_configured"},
                  {"status": "budget_exhausted"}, {"status": "too_long"}, {"status": "empty"})

    def slots(self, city=(), date=(), event_name=()):
        return {"city": list(city), "date": list(date), "event_name": list(event_name)}

    def test_everything_known_is_ready_to_search(self):
        slots = self.slots(city=["NYC"], date=["sat"], event_name=["musicals"])
        self.assertEqual(extract.follow_ups(slots, self.OK_MUSIC_TWO), [])

    def test_missing_city_and_date_asks_city_then_date(self):
        self.assertEqual(extract.follow_ups(self.slots(event_name=["x"]), self.OK_MUSIC_TWO), ["ask_city", "ask_date"])

    def test_missing_event_with_unknown_type_asks_what(self):
        slots = self.slots(city=["NYC"], date=["sat"])
        self.assertEqual(extract.follow_ups(slots, self.OK_UNKNOWN_TWO), ["ask_what"])

    def test_missing_event_with_failed_jev_asks_what_and_not_people(self):
        slots = self.slots(city=["NYC"], date=["sat"])
        for jev_result in self.FAILED_JEV:
            with self.subTest(jev_result=jev_result):
                self.assertEqual(extract.follow_ups(slots, jev_result), ["ask_what"])

    def test_known_event_with_unknown_type_does_not_ask_what(self):
        slots = self.slots(city=["NYC"], date=["sat"], event_name=["musicals"])
        self.assertEqual(extract.follow_ups(slots, self.OK_UNKNOWN_TWO), [])

    def test_ok_jev_with_unknown_tickets_asks_people(self):
        slots = self.slots(city=["NYC"], date=["sat"], event_name=["lizzo"])
        self.assertEqual(extract.follow_ups(slots, self.OK_MUSIC_UNKNOWN_TICKETS), ["ask_people"])

    def test_failed_jev_never_asks_people(self):
        slots = self.slots(city=["NYC"], date=["sat"], event_name=["lizzo"])
        for jev_result in self.FAILED_JEV:
            with self.subTest(jev_result=jev_result):
                self.assertEqual(extract.follow_ups(slots, jev_result), [])

    def test_every_question_in_order(self):
        jev_result = {"status": "ok", "event_type": "unknown", "tickets": "unknown"}
        self.assertEqual(extract.follow_ups(self.slots(), jev_result),
                         ["ask_city", "ask_date", "ask_what", "ask_people"])

    def test_every_question_when_jev_failed(self):
        self.assertEqual(extract.follow_ups(self.slots(), {"status": "error", "message": "x"}),
                         ["ask_city", "ask_date", "ask_what"])


class CompareTest(unittest.TestCase):
    def test_one_row_per_model_with_its_own_slots_and_follow_ups(self):
        def tagger(wanted):
            return lambda tokens: [f"B-{wanted[t]}" if t in wanted else "O" for t in tokens]

        models = [extract.LoadedModel("dictionary", "Dictionary", "baseline", tagger({"tmrw": "date"})),
                  extract.LoadedModel("bertweet", "BERTweet", "transformer",
                                   tagger({"lakers": "event_name", "tmrw": "date", "boston": "city"}))]
        rows = extract.compare("boston lakers tmrw", models, {"status": "skipped"})
        self.assertEqual([row["name"] for row in rows], ["Dictionary", "BERTweet"])
        self.assertEqual(rows[1]["slots"], {"city": ["boston"], "date": ["tmrw"], "event_name": ["lakers"]})
        self.assertEqual(rows[1]["spans"][0], {"slot": "city", "start": 0, "end": 6})
        self.assertEqual(rows[0]["follow_ups"], ["ask_city", "ask_what"])
        self.assertEqual(rows[1]["follow_ups"], [])


class LoadModelsTest(unittest.TestCase):
    def test_missing_artifacts_are_skipped_with_a_note(self):
        with tempfile.TemporaryDirectory() as empty:
            models, notes = extract.load_models(classic_dir=empty, artifacts_dir=empty)
        self.assertEqual(models, [])
        self.assertEqual(len(notes), len(extract.CLASSIC_MODELS) + len(extract.TRANSFORMER_MODELS))
        self.assertIn("BERTweet skipped", notes[-1])

    def test_transformers_are_skipped_with_a_note_when_no_gpu_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            for key, _ in extract.TRANSFORMER_MODELS:
                (Path(tmp) / key).mkdir()
                (Path(tmp) / key / "config.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(extract, "transformer_problem", return_value="no CUDA GPU is available"):
                models, notes = extract.load_models(classic_dir=tmp, artifacts_dir=tmp)
        self.assertEqual(models, [])
        self.assertEqual(notes[-3:], [f"{name} skipped: no CUDA GPU is available"
                                      for _, name in extract.TRANSFORMER_MODELS])

    def test_dictionary_tagger_data_becomes_a_tag_function(self):
        tagger = DictionaryTagger([{"tokens": ["Paris"], "tags": ["B-city"]}])
        tag = extract.classic_tagger({"model": "dictionary baseline", "tagger": tagger})
        self.assertEqual(tag(["paris", "x"]), ["B-city", "O"])


if __name__ == "__main__":
    unittest.main()
