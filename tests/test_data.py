"""Meaningful source-boundary, cleanup, extraction and split regression tests."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from classifier import LABELS
import prepare_data


class AnnotationTests(unittest.TestCase):
    def test_offsets_are_code_points_and_raw_newlines_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "article1.task2-TC.labels"
            text = "😀\r\nAlarm!"
            path.write_bytes(f"1\t{LABELS[0]}\t3\t9\n".encode("utf-8"))
            rows = prepare_data.read_annotation_file(path, "1", text)
            self.assertEqual(text[rows[0]["start"]:rows[0]["end"]], "Alarm!")

    def test_readable_validation_errors(self):
        cases = [
            (f"1\t{LABELS[0]}\t-1\t2", "offsets"),
            (f"1\t{LABELS[0]}\t0\t100", "offsets"),
            (f"1\t{LABELS[0]}\t1\t1", "offsets"),
            (f"1\t{LABELS[0]}\tx\t2", "integers"),
            (f"2\t{LABELS[0]}\t0\t2", "article ID"),
            ("1\tMade_Up\t0\t2", "unknown technique"),
            ("1\tthree\tfields", "four tab-separated"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotations.labels"
            for annotation, message in cases:
                with self.subTest(annotation=annotation):
                    path.write_text(annotation + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        prepare_data.read_annotation_file(path, "1", "not false!")

    def test_cleaning_preserves_negation_punctuation_and_unicode(self):
        self.assertEqual(prepare_data.normalize_whitespace("\tDo\u00a0not\nagree—ever! 😀 "),
                         "Do not agree—ever! 😀")
        text = "a" * 900 + "not false!" + "b" * 900
        context = prepare_data.make_context(text, 900, 910)
        self.assertIn("not false!", context)
        self.assertEqual(len(context), 710)
        self.assertLessEqual(len(context), 1500)

    def test_duplicate_removal_does_not_drop_other_labels_on_same_span(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            (raw / "articles").mkdir()
            (raw / "annotations").mkdir()
            (raw / "articles" / "article1.txt").write_bytes(b"Not\nfalse!")
            annotations = [f"1\t{LABELS[0]}\t0\t10", f"1\t{LABELS[0]}\t0\t10",
                           f"1\t{LABELS[1]}\t0\t10"]
            (raw / "annotations" / "article1.task2-TC.labels").write_text(
                "\n".join(annotations) + "\n", encoding="utf-8")
            rows, audit, hashes = prepare_data.build_rows(raw)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["text"], "Not\nfalse!")
            self.assertEqual(rows[0]["clean_text"], "Not false!")
            self.assertEqual(audit["source_annotation_rows"], 3)
            self.assertEqual(audit["exact_duplicate_annotations_removed"], 1)
            self.assertEqual(audit["multiple_label_source_spans"], 1)
            self.assertEqual(audit["removed_annotations"][0]["source_line"], 2)
            self.assertEqual(audit["supervised_articles"], 1)
            self.assertEqual(hashes["1"], hashlib.sha256(b"Not\nfalse!").hexdigest())

    def test_missing_article_pair_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            (raw / "articles").mkdir()
            (raw / "annotations").mkdir()
            (raw / "articles" / "article1.txt").write_bytes(b"No annotation")
            with self.assertRaisesRegex(ValueError, "corresponding annotation"):
                prepare_data.build_rows(raw)


class ExtractionTests(unittest.TestCase):
    def archive(self, path, extra=None):
        members = {
            "datasets/README.md": b"Original source README",
            "datasets/train-articles/article1.txt": b"example",
            "datasets/train-labels-task2-technique-classification/article1.task2-TC.labels": b"labels",
            "datasets/dev-articles/article2.txt": b"unused dev article",
            "../../outside.txt": b"must not be extracted",
        }
        with tarfile.open(path, "w:gz") as handle:
            for name, body in members.items():
                member = tarfile.TarInfo(name)
                member.size = len(body)
                handle.addfile(member, io.BytesIO(body))
            if extra is not None:
                handle.addfile(extra)

    def test_checksum_precedes_tar_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corrupt.tgz"
            path.write_bytes(b"not a tar archive")
            with patch("prepare_data.tarfile.open") as opened:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    prepare_data.extract_archive(path, Path(temporary) / "output")
                opened.assert_not_called()

    def test_only_whitelisted_regular_files_are_extracted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.tgz"
            output = Path(temporary) / "output"
            self.archive(path)
            with patch("prepare_data.SHA256", prepare_data.digest(path)):
                prepare_data.extract_archive(path, output)
            files = sorted(str(item.relative_to(output)).replace("\\", "/")
                           for item in output.rglob("*") if item.is_file())
            self.assertEqual(files, ["datasets/README.md", "datasets/train-articles/article1.txt",
                                    "datasets/train-labels-task2-technique-classification/article1.task2-TC.labels"])
            self.assertFalse((Path(temporary).parent / "outside.txt").exists())

    def test_whitelisted_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.tgz"
            link = tarfile.TarInfo("datasets/train-articles/article3.txt")
            link.type = tarfile.SYMTYPE
            link.linkname = "/outside"
            self.archive(path, link)
            with patch("prepare_data.SHA256", prepare_data.digest(path)):
                with self.assertRaisesRegex(ValueError, "regular files"):
                    prepare_data.extract_archive(path, Path(temporary) / "output")


class SplitTests(unittest.TestCase):
    def fixture(self):
        return [{"id": f"{article}:{label}", "article_id": str(article),
                 "text": f"{label} example in article {article}",
                 "context": f"article {article} context", "label": label}
                for article in range(20) for label in LABELS]

    def test_split_is_deterministic_article_grouped_and_label_independent(self):
        rows = self.fixture()
        manifest = prepare_data.assign_partitions(rows)
        other = self.fixture()
        self.assertEqual(manifest, prepare_data.assign_partitions(other))
        self.assertEqual(manifest["article_counts"], {"train": 13, "dev": 3, "test": 4})
        for article_id in manifest["article_partitions"]:
            self.assertEqual(len({row["partition"] for row in rows if row["article_id"] == article_id}), 1)
        # Permuting labels cannot change the assignment; every article has all labels.
        permuted = self.fixture()
        for row in permuted:
            row["label"] = LABELS[(LABELS.index(row["label"]) + 1) % len(LABELS)]
        self.assertEqual(manifest["article_partitions"],
                         prepare_data.assign_partitions(permuted)["article_partitions"])
        self.assertFalse(manifest["official_test_comparable"])

    def test_identical_article_hashes_and_model_inputs_are_grouped(self):
        rows = self.fixture()
        hashes = {str(article): str(article) for article in range(20)}
        hashes["1"] = hashes["0"]
        first, second = rows[2 * len(LABELS)], rows[3 * len(LABELS)]
        second["text"] = first["text"].upper().replace(" ", "\u00a0")
        second["context"] = first["context"].upper()
        manifest = prepare_data.assign_partitions(rows, article_hashes=hashes)
        self.assertEqual(manifest["group_count"], 18)
        assignment = manifest["article_partitions"]
        self.assertEqual(assignment["0"], assignment["1"])
        self.assertEqual(assignment["2"], assignment["3"])

    def test_train_must_contain_all_labels(self):
        rows = self.fixture()
        for row in rows:
            row["label"] = LABELS[0]
        with self.assertRaisesRegex(ValueError, "training split lacks techniques"):
            prepare_data.assign_partitions(rows)

    def test_small_or_noninteger_seed_fails(self):
        with self.assertRaisesRegex(ValueError, "independent article groups"):
            prepare_data.assign_partitions(self.fixture()[:len(LABELS)])
        with self.assertRaisesRegex(ValueError, "integer"):
            prepare_data.assign_partitions(self.fixture(), seed=True)


class DevelopmentDataTests(unittest.TestCase):
    def fixture(self, directory):
        directory = Path(directory)
        (directory / "raw" / "articles").mkdir(parents=True)
        (directory / "raw" / "annotations").mkdir(parents=True)
        (directory / "cleaned").mkdir()
        training = "😀\r\nTraining source. " + " ".join(f"clause {i} not false!" for i in range(len(LABELS)))
        annotations = []
        for index, label in enumerate(LABELS):
            excerpt = f"clause {index} not false!"
            start = training.index(excerpt)
            annotations.append(f"1\t{label}\t{start}\t{start + len(excerpt)}")
        annotations.append(annotations[0])
        dev = "😀\r\nDevelopment source. " + "a" * 1500 + "not\tfalse!" + "b" * 1500
        start = dev.index("not\tfalse!")
        (directory / "raw" / "articles" / "article1.txt").write_bytes(training.encode("utf-8"))
        (directory / "raw" / "articles" / "article2.txt").write_bytes(dev.encode("utf-8"))
        (directory / "raw" / "annotations" / "article1.task2-TC.labels").write_text(
            "\n".join(annotations) + "\n", encoding="utf-8")
        (directory / "raw" / "annotations" / "article2.task2-TC.labels").write_text(
            f"2\t{LABELS[0]}\t{start}\t{start + 10}\n", encoding="utf-8")
        manifest = {"seed": 42, "source_partition": "official_train",
                    "article_partitions": {"1": "train", "2": "dev", "3": "test"},
                    "row_counts": {"train": len(LABELS), "dev": 1, "test": 1}}
        prepare_data.write_json(directory / "cleaned" / "splits.json", manifest)
        (directory / "cleaned" / "techniques.jsonl").write_bytes(b"original benchmark sentinel\n")
        (directory / "cleaned" / "audit.json").write_bytes(b"original audit sentinel\n")
        return directory, training, dev, start

    def test_wider_context_preserves_raw_offsets_partitions_and_original_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, training, dev, start = self.fixture(temporary)
            original = {name: (directory / "cleaned" / name).read_bytes()
                        for name in ("techniques.jsonl", "audit.json", "splits.json")}
            rows, audit = prepare_data.build_development_rows(directory)
            self.assertEqual(audit["row_counts"], {"train": 14, "dev": 1})
            self.assertEqual(audit["exact_duplicate_annotations_removed"], 1)
            self.assertEqual({row["partition"] for row in rows}, {"train", "dev"})
            self.assertEqual(audit["context_margin"], 1000)
            dev_row = next(row for row in rows if row["partition"] == "dev")
            self.assertEqual(dev_row["text"], "not\tfalse!")
            self.assertEqual(dev_row["clean_text"], "not false!")
            self.assertEqual(dev_row["context"], prepare_data.normalize_whitespace(dev[start - 1000:start + 1010]))
            self.assertGreater(len(dev_row["context"]), 1500)
            for row in rows:
                source = training if row["article_id"] == "1" else dev
                self.assertEqual(row["text"], source[row["start"]:row["end"]])
                self.assertEqual(row["id"], f"{row['article_id']}:{row['start']}:{row['end']}:{row['label']}")
                self.assertEqual(row["source_partition"], "official_train")
            self.assertEqual(original, {name: (directory / "cleaned" / name).read_bytes() for name in original})

    def test_context_at_article_edges_and_emoji_remain_code_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, training, _, _ = self.fixture(temporary)
            rows, _ = prepare_data.build_development_rows(directory)
            first = next(row for row in rows if row["article_id"] == "1")
            self.assertEqual(first["context"], prepare_data.normalize_whitespace(training))
            self.assertTrue(first["context"].startswith("😀 Training source."))
            self.assertEqual(training[first["start"]:first["end"]], first["text"])

    def test_test_source_files_are_never_opened(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, _, _, _ = self.fixture(temporary)
            original_read = Path.read_bytes
            opened = []

            def checked_read(path):
                opened.append(path.name)
                if path.name.startswith("article3."):
                    raise AssertionError("A test source was opened.")
                return original_read(path)

            with patch.object(Path, "read_bytes", checked_read):
                rows, audit = prepare_data.build_development_rows(directory)
            self.assertNotIn("article3.txt", opened)
            self.assertNotIn("article3.task2-TC.labels", opened)
            self.assertEqual(audit["test_source_files_opened"], 0)
            self.assertEqual(audit["test_rows_loaded"], 0)
            self.assertEqual(len(audit["source_file_sha256"]), 4)
            self.assertNotIn("3", {row["article_id"] for row in rows})

    def test_changed_partition_counts_and_unsafe_ids_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, _, _, _ = self.fixture(temporary)
            path = directory / "cleaned" / "splits.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["row_counts"]["dev"] = 2
            prepare_data.write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "row counts differ"):
                prepare_data.build_development_rows(directory)
            manifest["article_partitions"]["../unsafe"] = "train"
            prepare_data.write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "ASCII digits"):
                prepare_data.build_development_rows(directory)

    def test_identical_wider_inputs_cannot_cross_train_and_dev(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, training, _, _ = self.fixture(temporary)
            article = directory / "raw" / "articles" / "article2.txt"
            article.write_bytes(training.encode("utf-8"))
            start = training.index("clause 0 not false!")
            (directory / "raw" / "annotations" / "article2.task2-TC.labels").write_text(
                f"2\t{LABELS[0]}\t{start}\t{start + len('clause 0 not false!')}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cross train/dev"):
                prepare_data.build_development_rows(directory)


if __name__ == "__main__":
    unittest.main()
