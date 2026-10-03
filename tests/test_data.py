import csv
import itertools
import unittest
from collections import defaultdict
from pathlib import Path

from prepare_data import LABELS, normalize, prepare


def event(title, description, category="Konser", **fields):
    return {
        "title": title,
        "description": description,
        "category": category,
        **fields,
    }


class NormalizeTests(unittest.TestCase):
    def test_normalize_uses_turkish_i_and_keeps_turkish_letters(self):
        self.assertEqual(
            normalize("  IĞDIR, İSTANBUL! ÇAĞRI ŞÖLENİ  "),
            "ığdır istanbul çağrı şöleni",
        )

    def test_normalize_collapses_case_punctuation_and_space(self):
        self.assertEqual(normalize("Rock—JAZZ\tGecesi"), "rock jazz gecesi")


class PrepareTests(unittest.TestCase):
    def test_repeated_sessions_keep_one_text_example(self):
        records = [
            event(
                "Aynı Konser",
                "Canlı müzik programı",
                source="first",
                url="https://tickets.example/events/show?session=1",
            ),
            event(
                "AYNI KONSER",
                "Canlı müzik programı",
                source="second",
                url="https://tickets.example/events/show?session=2",
            ),
        ]

        rows, audit = prepare(records)

        self.assertEqual(len(rows), 1)
        self.assertEqual(audit["unique_examples"], 1)
        self.assertEqual(audit["event_families"], 1)
        self.assertEqual(audit["excluded_records"]["duplicate_session_records"], 1)

    def test_connected_keys_form_one_transitive_family(self):
        records = [
            event("Ortak Başlık", "Birinci canlı sahne metni"),
            event(
                "ORTAK BAŞLIK",
                "İkinci farklı sahne metni",
                productionKey="production-7",
            ),
            event(
                "Başka Başlık",
                "Üçüncü farklı sahne metni",
                productionKey="production-7",
            ),
        ]

        rows, audit = prepare(records)

        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["group_id"] for row in rows}), 1)
        self.assertEqual(audit["event_families"], 1)

    def test_long_exact_description_and_canonical_page_url_connect_records(self):
        shared_description = "sekiz ayrı sözcük içeren aynı uzun etkinlik açıklaması burada"
        records = [
            event("İlk Başlık", shared_description),
            event("İkinci Başlık", shared_description),
            event(
                "Üçüncü Başlık",
                "Bu da farklı ve yeterince uzun bir tanıtım metnidir",
                url="https://tickets.example/show/3/?utm_source=a#top",
            ),
            event(
                "Dördüncü Başlık",
                "Tamamen başka ve yeterince uzun olan tanıtım metnidir",
                url="https://tickets.example/show/3?session=evening",
            ),
        ]

        rows, audit = prepare(records)

        groups_by_title = {row["title"]: row["group_id"] for row in rows}
        self.assertEqual(groups_by_title["İlk Başlık"], groups_by_title["İkinci Başlık"])
        self.assertEqual(groups_by_title["Üçüncü Başlık"], groups_by_title["Dördüncü Başlık"])
        self.assertEqual(audit["event_families"], 2)

    def test_conflicting_provider_labels_exclude_the_whole_family(self):
        records = [
            event("Türü Belirsiz Gösteri", "Birinci açıklama metni", "Konser"),
            event("TÜRÜ BELİRSİZ GÖSTERİ", "İkinci açıklama metni", "Tiyatro"),
            event("Bağımsız Oyun", "Tek kişilik tiyatro oyunu", "Tiyatro"),
        ]

        rows, audit = prepare(records)

        self.assertEqual([row["title"] for row in rows], ["Bağımsız Oyun"])
        self.assertEqual(audit["excluded_records"]["conflicting_family_records"], 2)
        self.assertEqual(audit["class_counts"], {"theatre": 1})

    def test_unsupported_labels_and_non_records_are_rejected(self):
        records = [
            event("Film Gösterimi", "Uzun bir sinema gösterimi", "Sinema"),
            event("Eksik Etiket", "Uzun bir etkinlik açıklaması", None),
            "not a record",
            event("Geçerli Stand Up", "Komedi gecesi gösterisi", "Stand-up"),
        ]

        rows, audit = prepare(records)

        self.assertEqual([row["label"] for row in rows], ["stand_up"])
        self.assertEqual(audit["excluded_records"]["unsupported_label"], 3)
        self.assertEqual(audit["input_session_records"], 4)

    def test_short_and_empty_text_are_excluded(self):
        records = [
            event("", ""),
            event("İki Kelime", "İki Kelime"),
            event("Üç Kelimelik Etkinlik", "Üç Kelimelik Etkinlik"),
        ]

        rows, audit = prepare(records, min_words=3)

        self.assertEqual([row["text"] for row in rows], ["Üç Kelimelik Etkinlik"])
        self.assertEqual(audit["excluded_records"]["short_text_records"], 2)

    def test_output_is_deterministic_for_input_permutations(self):
        records = [
            event(
                "Tekrarlanan Konser",
                "Canlı müzik etkinliği",
                source="z-provider",
                url="https://z.example/session/2",
            ),
            event(
                "TEKRARLANAN KONSER",
                "Canlı müzik etkinliği",
                source="a-provider",
                url="https://a.example/session/1",
            ),
            event("Bağımsız Tiyatro", "Yeni sahne oyunu", "Tiyatro"),
        ]
        expected_rows, expected_audit = prepare(records)

        for permutation in itertools.permutations(records):
            with self.subTest(order=[record.get("source", "") for record in permutation]):
                rows, audit = prepare(list(permutation))
                self.assertEqual(rows, expected_rows)
                self.assertEqual(audit, expected_audit)


class FrozenDatasetTests(unittest.TestCase):
    def test_real_csv_has_unique_normalized_inputs_and_homogeneous_groups(self):
        path = Path(__file__).parents[1] / "data" / "events.csv"
        if not path.exists():
            self.skipTest("data/events.csv is a local ignored artifact")

        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))

        self.assertGreaterEqual(len(rows), 1000)
        self.assertEqual({row["label"] for row in rows}, set(LABELS.values()))
        normalized_inputs = [normalize(row["text"]) for row in rows]
        self.assertEqual(len(normalized_inputs), len(set(normalized_inputs)))
        labels_by_group = defaultdict(set)
        for row in rows:
            self.assertTrue(row["group_id"])
            labels_by_group[row["group_id"]].add(row["label"])
        self.assertTrue(all(len(labels) == 1 for labels in labels_by_group.values()))


if __name__ == "__main__":
    unittest.main()
