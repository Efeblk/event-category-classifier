import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from classifier import OUTPUT_LABELS
from jev import CRITERIA, JevClient, load_settings, validate_jev_response


def response(label="Loaded_Language"):
    return {"model": "jev-1.13.0", "answers": {"activity": {
        "type": "choice", "choice": label, "confidence": 0.8,
        "probabilities": {key: 0.85 if key == label else 0.15 / (len(OUTPUT_LABELS)-1) for key in OUTPUT_LABELS}}},
        "usage": {"input_tokens": 100, "output_tokens": 20}}


class JevClientTests(unittest.TestCase):
    def test_valid_provider_answer_is_accepted(self):
        self.assertEqual(validate_jev_response(response(), "jev-1.13.0")["choice"], "Loaded_Language")

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

    def test_live_request_has_same_text_and_15_labels_but_no_key_in_body_or_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = response()
            provider["debug"] = "private-test-key"
            provider["usage"]["debug"] = "private-test-key"
            transport = Mock(return_value=provider)
            client = JevClient({"TYPESAFE_API_KEY": "private-test-key", "JEV_MAX_CALLS": "1"}, directory, transport)
            result = client.classify("Those crooked officials", context="Those crooked officials betrayed us.")
            self.assertEqual(result["label"], "Loaded_Language")
            self.assertNotIn("private-test-key", json.dumps(result))
            body = transport.call_args.args[0]
            payload = json.loads(body)
            self.assertEqual(payload["state"], {"text": "Those crooked officials", "context": "Those crooked officials betrayed us."})
            self.assertEqual(set(payload["questions"]["activity"]["criteria"]), set(OUTPUT_LABELS))
            self.assertNotIn(b"private-test-key", body)
            receipt = (Path(directory) / "001.json").read_text()
            self.assertNotIn("private-test-key", receipt)
            self.assertNotIn("crooked", receipt)

    def test_unicode_payload_fits_original_byte_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = Mock(return_value=response())
            client = JevClient({"TYPESAFE_API_KEY":"fixture", "JEV_MAX_CALLS":"1"}, directory, transport)
            self.assertEqual(client.classify("ş" * 1000)["status"],"ok")
            self.assertLessEqual(len(transport.call_args.args[0]),8000)
            self.assertEqual(len(json.loads(transport.call_args.args[0])["questions"]["activity"]["criteria"]),15)

    def test_disabled_and_oversized_requests_make_no_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = Mock()
            disabled = JevClient({}, directory, transport)
            self.assertEqual(disabled.classify("play_music")["status"], "not_configured")
            enabled = JevClient({"TYPESAFE_API_KEY": "test", "JEV_MAX_CALLS": "1"}, directory, transport)
            self.assertEqual(enabled.classify("x" * 3001)["status"], "not_run")
            self.assertEqual(enabled.classify("😀" * 3000)["status"], "not_run")
            self.assertEqual(enabled.remaining_calls(), 1)
            transport.assert_not_called()

    def test_preflight_preserves_context_above_former_cap_without_a_call(self):
        # Representative length above the former cap; context must stay unchanged.
        client = JevClient({}, transport=Mock())
        payload = json.loads(client.prepare_request("fear", "c" * 1490))
        self.assertEqual(payload["state"], {"text": "fear", "context": "c" * 1490})
        client.transport.assert_not_called()

    def test_invalid_excerpt_or_context_is_rejected_before_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = Mock()
            client = JevClient({"TYPESAFE_API_KEY": "fixture", "JEV_MAX_CALLS": "1"}, directory, transport)
            for text, context in (("", ""), (None, ""), ("hello", [])):
                with self.assertRaises(ValueError):
                    client.classify(text, context)
            self.assertEqual(client.remaining_calls(), 1)
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
