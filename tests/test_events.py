import unittest

from classifier import LABELS
from prepare_events import CATEGORY_TO_LABEL, LABELS as PREPARE_LABELS, build_rows, clean_title


def event(event_id, name, category, performers="['p1']"):
    return ("events/data/day.csv", {"eventId": event_id, "name": name, "category": category,
                                    "performerIds": performers})


class EventCorpusTests(unittest.TestCase):
    def test_label_copy_matches_classifier(self):
        self.assertEqual(PREPARE_LABELS, LABELS)
        self.assertEqual(set(CATEGORY_TO_LABEL.values()), set(LABELS))

    def test_ticketing_notes_are_removed_but_title_words_stay(self):
        self.assertEqual(clean_title("Nurse John (18+ Event)"), "Nurse John")
        self.assertEqual(clean_title("Damon Darling (Rescheduled from 1/25, 5/10)"), "Damon Darling")
        self.assertEqual(clean_title("Dear Evan Hansen (Open Caption)"), "Dear Evan Hansen")
        self.assertEqual(clean_title("Taylor Tomlinson (Tries Out New Ideas)"),
                         "Taylor Tomlinson (Tries Out New Ideas)")

    def test_categories_are_mapped_and_other_categories_dropped(self):
        rows, audit = build_rows([
            event("1", "Metallica", "music", "['a']"),
            event("2", "Hamilton", "theater", "['b']"),
            event("3", "Kevin Hart", "comedy", "['c']"),
            event("4", "Cubs at Brewers", "mlb", "['d']"),
            event("5", "Toronto International Film Festival - Masc", "theater", "['e']"),
        ])
        self.assertEqual({row["text"]: row["label"] for row in rows},
                         {"Metallica": "concert", "Hamilton": "theatre", "Kevin Hart": "stand_up"})
        self.assertEqual(audit["license"], "CC BY-NC 4.0")
        self.assertEqual(audit["dropped_film_festival_events"], 1)

    def test_repeated_dates_become_one_row_and_conflicting_titles_are_dropped(self):
        rows, audit = build_rows([
            event("1", "Elf The Musical", "theater", "['a']"),
            event("2", "Elf The Musical (21+ Event)", "theater", "['a']"),
            event("3", "Death Lens", "music", "['b']"),
            event("4", "Death Lens", "comedy", "['c']"),
        ])
        self.assertEqual([row["text"] for row in rows], ["Elf The Musical"])
        self.assertEqual(audit["dropped_conflicting_titles"], 1)

    def test_one_performer_keeps_one_group_and_mixed_groups_are_dropped(self):
        rows, audit = build_rows([
            event("1", "Taylor Tomlinson", "comedy", "['t']"),
            event("2", "Taylor Tomlinson - New Hour", "comedy", "['t']"),
            event("3", "Nick Cannon - Wild N Out", "comedy", "['n']"),
            event("4", "Nick Cannon Live", "music", "['n']"),
        ])
        self.assertEqual({row["group_id"] for row in rows}, {"t"})
        self.assertEqual(len(rows), 2)
        self.assertEqual(audit["dropped_rows_in_mixed_label_groups"], 2)


if __name__ == "__main__":
    unittest.main()
