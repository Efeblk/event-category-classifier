import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prepare_massive import digest, extract_archive, parse_annotation


class MassiveTests(unittest.TestCase):
    def test_bio_and_suffixes(self):
        self.assertEqual(
            parse_annotation("yarın toplantıyla", "[date : yarın] [event_name : toplantıyla]"),
            (["yarın", "toplantıyla"], ["B-date", "B-event_name"]),
        )
        self.assertEqual(parse_annotation("bu hafta gel", "[date : bu hafta] gel")[1], ["B-date", "I-date", "O"])

    def test_invalid_annotation(self):
        cases = [
            ("yarın", "bugün"),  # text does not match
            ("ankara'da", "[place_name : ankara]'da"),  # boundary inside a word
        ]
        for text, annotation in cases:
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                parse_annotation(text, annotation)

    def test_hash_and_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "a.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for name in ("1.0/data/tr-TR.jsonl", "1.0/LICENSE", "../outside"):
                    info = tarfile.TarInfo(name)
                    info.size = 1
                    tar.addfile(info, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "sha256"):
                extract_archive(archive, Path(tmp) / "out")
            with patch("prepare_massive.SHA256", digest(archive)):
                extract_archive(archive, Path(tmp) / "out")
            self.assertFalse((Path(tmp) / "outside").exists())
            self.assertTrue((Path(tmp) / "out/1.0/LICENSE").exists())


if __name__ == "__main__":
    unittest.main()
