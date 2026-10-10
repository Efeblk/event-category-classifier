"""Tests for jev.py with a fake transport: no network, no real API key, receipts in a temporary directory."""

import json
import tempfile
import unittest
from pathlib import Path

import jev

MODEL = "jev-1.13.0"


def good_response():
    return {
        "model": MODEL,
        "answers": {
            "event_type": {"type": "choice", "choice": "music", "confidence": 0.9,
                           "probabilities": {"music": 0.8, "sports": 0.1, "theater": 0.05,
                                             "comedy": 0.02, "festival": 0.02, "unknown": 0.01}},
            "tickets": {"type": "choice", "choice": "two", "confidence": 0.7,
                        "probabilities": {"one": 0.1, "two": 0.7, "three_four": 0.1,
                                          "five_plus": 0.05, "unknown": 0.05}},
        },
    }


class FakeTransport:
    """Records each request body and returns a copy of a fixed response or raises a fixed error."""

    def __init__(self, response=None, error=None):
        self.response, self.error, self.bodies = response, error, []

    def __call__(self, body):
        self.bodies.append(body)
        if self.error is not None:
            raise self.error
        return json.loads(json.dumps(self.response))


class JevTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ledger = Path(tmp.name) / "jev_calls"

    def client(self, transport, max_calls=2, key="test-key"):
        settings = {"TYPESAFE_API_KEY": key, "JEV_MAX_CALLS": str(max_calls)}
        return jev.JevClient(settings=settings, ledger_dir=self.ledger, transport=transport)


class ValidAnswerTest(JevTestCase):
    def test_valid_answer_is_parsed_and_receipted(self):
        transport = FakeTransport(good_response())
        result = self.client(transport).classify("with my girlfriend, a concert tomorrow")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["event_type"], "music")
        self.assertEqual(result["tickets"], "two")
        self.assertEqual(result["confidence"], {"event_type": 0.9, "tickets": 0.7})
        self.assertEqual(len(transport.bodies), 1)
        body = json.loads(transport.bodies[0])
        self.assertEqual(set(body["questions"]), {"event_type", "tickets"})
        self.assertEqual(body["state"]["text"], "with my girlfriend, a concert tomorrow")
        receipt = json.loads((self.ledger / "001.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "ok")
        self.assertNotIn("text", receipt)


class RejectionTest(JevTestCase):
    def assert_rejected(self, response):
        transport = FakeTransport(response)
        result = self.client(transport).classify("some request")
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(transport.bodies), 1)
        receipt = json.loads((self.ledger / "001.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "error")

    def test_choice_outside_allowed_labels_is_rejected(self):
        response = good_response()
        response["answers"]["event_type"]["choice"] = "rock"
        self.assert_rejected(response)

    def test_probabilities_must_sum_to_one(self):
        response = good_response()
        response["answers"]["tickets"]["probabilities"] = {"one": 0.1, "two": 0.2, "three_four": 0.1,
                                                           "five_plus": 0.05, "unknown": 0.05}
        self.assert_rejected(response)

    def test_wrong_model_is_rejected(self):
        response = good_response()
        response["model"] = "jev-9.9.9"
        self.assert_rejected(response)

    def test_transport_failure_is_not_retried(self):
        transport = FakeTransport(error=OSError("network down"))
        result = self.client(transport).classify("some request")
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(transport.bodies), 1)


class BudgetTest(JevTestCase):
    def test_limit_stops_calls_after_n_receipts(self):
        transport = FakeTransport(good_response())
        client = self.client(transport, max_calls=2)
        statuses = [client.classify(f"request {i}")["status"] for i in range(3)]
        self.assertEqual(statuses, ["ok", "ok", "budget_exhausted"])
        self.assertEqual(len(transport.bodies), 2)
        self.assertTrue((self.ledger / "001.json").exists())
        self.assertTrue((self.ledger / "002.json").exists())
        self.assertFalse((self.ledger / "003.json").exists())

    def test_existing_receipts_count_toward_the_limit(self):
        self.ledger.mkdir(parents=True)
        (self.ledger / "001.json").write_text("{}", encoding="utf-8")
        transport = FakeTransport(good_response())
        self.assertEqual(self.client(transport, max_calls=1).classify("hi")["status"], "budget_exhausted")
        self.assertEqual(transport.bodies, [])


class NoCallTest(JevTestCase):
    def test_missing_key_makes_no_call(self):
        transport = FakeTransport(good_response())
        result = self.client(transport, key="").classify("hi")
        self.assertEqual(result["status"], "not_configured")
        self.assertEqual(transport.bodies, [])

    def test_zero_limit_disables_calls(self):
        transport = FakeTransport(good_response())
        self.assertEqual(self.client(transport, max_calls=0).classify("hi")["status"], "not_configured")
        self.assertEqual(transport.bodies, [])

    def test_long_input_is_not_sent(self):
        transport = FakeTransport(good_response())
        result = self.client(transport).classify("a" * 1001)
        self.assertEqual(result["status"], "too_long")
        self.assertEqual(transport.bodies, [])

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            self.client(FakeTransport(), max_calls=51)
        with self.assertRaises(ValueError):
            jev.JevClient(settings={"JEV_MODEL": "latest"}, ledger_dir=self.ledger)


if __name__ == "__main__":
    unittest.main()
