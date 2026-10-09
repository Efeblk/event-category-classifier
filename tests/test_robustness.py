"""Input and unlabeled-source checks without prepared data or live providers."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from method_comparison import METHODS, compare_request, validate_input
from stress_test import TEMPLATE, build_inputs, probe, variants


def source_fixture(path: Path, unsafe: bool = False, template: str | None = None) -> str:
    articles = {"101": "Intro \U0001f984\r\nHuman words with context.",
                "202": "Other news. Public words by a person."}
    annotations = []
    with tarfile.open(path, "w:gz") as archive:
        for article_id, article in articles.items():
            phrase = "Human words" if article_id == "101" else "Public words"
            start = article.index(phrase)
            annotations.append(f"{article_id}\t?\t{start}\t{start + len(phrase)}")
            raw = article.encode("utf-8")
            member = tarfile.TarInfo(f"datasets/dev-articles/article{article_id}.txt")
            if unsafe and article_id == "101":
                member.type, member.linkname = tarfile.SYMTYPE, "unrelated.txt"
                archive.addfile(member)
            else:
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        payload = (template if template is not None else "\n".join(annotations)).encode("utf-8")
        member = tarfile.TarInfo(TEMPLATE)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
        unrelated = tarfile.TarInfo("datasets/train-articles/article303.txt")
        unrelated.type, unrelated.linkname = tarfile.SYMTYPE, "must-not-read"
        archive.addfile(unrelated)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InputRobustnessTests(unittest.TestCase):
    def test_invalid_unicode_and_invisible_text_fail_before_models_or_provider(self):
        provider = Mock()
        with patch("method_comparison.validate_artifacts") as artifacts:
            for text in ("\ud800", "\udfff", "text\x00", "text\x1b", "\u200b\u200d", "\ufeff", " \u202e ", "\ufe0f", "\u034f"):
                with self.subTest(text=ascii(text)), self.assertRaises(ValueError):
                    compare_request({}, text, provider, True)
            for context in ("\ud800", "words\x00", "\u200b"):
                with self.subTest(context=ascii(context)), self.assertRaises(ValueError):
                    compare_request({}, "Human words", provider, True, context)
        artifacts.assert_not_called()
        provider.classify.assert_not_called()

    def test_visible_unicode_emoji_formatting_and_line_breaks_remain_exact(self):
        text, context = "Don't change cafe\u0301 \U0001f468\u200d\U0001f469 \u2600\ufe0f\n\ttext!", "More\r\ncontext \u200bhere."
        validate_input(text, context)
        with patch("method_comparison.validate_artifacts", return_value={m: {"model_name": m} for m in METHODS}), \
             patch("method_comparison.classify_request", return_value={"label": "Doubt"}) as learned:
            compare_request({}, text, context=context)
        for call in learned.call_args_list:
            self.assertEqual(call.args[1], text)
            self.assertEqual(call.kwargs["context"], context)

    def test_warnings_describe_actual_input_conditions_without_language_claims(self):
        with patch("method_comparison.validate_artifacts", return_value={m: {"model_name": m} for m in METHODS}), \
             patch("method_comparison.classify_request", return_value={"label": "unclear"}):
            plain = compare_request({}, "Human words", context="Nearby words.")
            missing = compare_request({}, "Human words")
            symbols = compare_request({}, "2026", context="Nearby words.")
        self.assertEqual(plain["warnings"], [])
        self.assertEqual(len(missing["warnings"]), 1)
        self.assertEqual(len(symbols["warnings"]), 1)
        self.assertIn("no letters", symbols["warnings"][0])


class HumanSourceTests(unittest.TestCase):
    def test_checksum_is_checked_before_opening_tar(self):
        with patch("stress_test.digest", return_value="bad"), patch("stress_test.tarfile.open") as opened:
            with self.assertRaisesRegex(ValueError, "checksum"):
                build_inputs("unused")
        opened.assert_not_called()

    def test_source_offsets_unicode_and_sampling_are_reproducible_and_unlabeled(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "fixture.tgz"
            checksum = source_fixture(archive)
            with patch("stress_test.SHA256", checksum):
                first = build_inputs(archive, 2)
                second = build_inputs(archive, 2)
        self.assertEqual(first, second)
        self.assertEqual({r["article_id"] for r in first}, {"101", "202"})
        self.assertEqual(first[0]["text"], "Human words")
        self.assertIn("\U0001f984", first[0]["context"])
        self.assertEqual(first[0]["start"], len("Intro \U0001f984\r\n"))
        self.assertTrue(all(r["gold"] is None and r["source_partition"] == "official_dev_unlabeled" for r in first))
        self.assertTrue(all(r["source_member"].startswith("datasets/dev-articles/") for r in first))

    def test_unsafe_members_gold_labels_and_bad_offsets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, unsafe, template in (
                ("symlink", True, None),
                ("gold", False, "101\tLoaded_Language\t0\t4"),
                ("negative", False, "101\t?\t-1\t4"),
                ("outside", False, "101\t?\t0\t9000"),
            ):
                with self.subTest(name=name):
                    archive = Path(directory) / (name + ".tgz")
                    checksum = source_fixture(archive, unsafe, template)
                    with patch("stress_test.SHA256", checksum), self.assertRaises(ValueError):
                        build_inputs(archive, 1)

    def test_probe_reports_changes_not_accuracy_and_never_calls_a_provider(self):
        rows = [{"id": "source:1", "article_id": "1", "text": "Human words", "context": "More words", "gold": None}]
        artifacts = {m: {"model_name": m, "dataset_sha256": "a" * 64, "code_sha256": "b" * 64,
                         "seed": 42, "provenance": {"evaluation_scope": "dev_only", "context_margin": 1000}}
                     for m in METHODS}

        def comparison(models, text, **kwargs):
            self.assertNotIn("include_jev", kwargs)
            return {"results": [{"method": method, "label": "unclear" if text in ("qxzvqxzv", "\U0001f984\U0001fa90\U0001f6f8") else "Doubt"}
                                for method in METHODS]}

        with patch("stress_test.validate_artifacts"), patch("stress_test.compare_request", side_effect=comparison):
            report = probe(artifacts, rows)
        self.assertIsNone(report["accuracy"])
        self.assertFalse(report["fit_performed"])
        self.assertEqual(report["paid_calls"], 0)
        self.assertEqual(report["training_seed"], 42)
        self.assertEqual(report["model_evaluation_scope"], "dev_only")
        self.assertEqual(report["context_margin"], 1000)
        self.assertEqual(report["diagnostic_case_count"], 8)
        self.assertEqual(report["transformed_case_count"], 7)
        self.assertTrue(all(row["passed"] is not False for row in report["controls"]))
        self.assertEqual(report["label_changes"]["whitespace"]["logistic_regression"]["count"], 0)
        self.assertIsNone(report["label_changes"]["curly_apostrophe"]["logistic_regression"]["fraction"])
        self.assertEqual(report["label_changes"]["curly_apostrophe"]["logistic_regression"]["pairs"], 0)
        self.assertNotIn("gold", report["comparisons"][0])
        self.assertEqual(rows[0]["text"], variants(rows[0])["original"][0])
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
