import unittest

from parser import parse_request, parser_scores


class ParserTests(unittest.TestCase):
    def test_clear_train_style_intents(self):
        cases = {
            "alarm kur": "alarm_set",
            "beni yarın uyandır": "alarm_set",
            "hava nasıl": "weather_query",
            "bir şarkı çal": "play_music",
            "etkinlikleri göster": "recommendation_events",
            "takvimime toplantı ekle": "calendar_set",
            "saat kaç": "datetime_query",
            "IŞIKLARI KAPAT": "iot_hue_lightoff",
        }
        for text, label in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_request(text), label)

    def test_unknown_ambiguous_and_negated_abstain(self):
        for text in ("merhaba", "alarm kur ve bir şarkı çal", "şarkı çalma", "ışığı kapatma", ""):
            with self.subTest(text=text):
                self.assertEqual(parse_request(text), "unclear")

    def test_abstention_metrics(self):
        result = parser_scores(["alarm_set"] * 3, ["alarm_set", "unclear", "play_music"])
        self.assertEqual(result, {
            "coverage": 2 / 3,
            "answered_accuracy": .5,
            "overall_accuracy": 1 / 3,
            "answered": 2,
            "correct": 1,
            "total": 3,
        })
        self.assertIsNone(parser_scores(["alarm_set"], ["unclear"])["answered_accuracy"])


if __name__ == "__main__":
    unittest.main()
