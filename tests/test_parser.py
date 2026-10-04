import unittest

from parser import guard_request, parse_request


class RequestBoundaryTests(unittest.TestCase):
    def test_multiple_positive_categories_need_clarification(self):
        for text in ("Konser veya tiyatro olabilir", "Find a concert or a theatre play"):
            self.assertEqual(guard_request(text)[1], "multiple_categories")

    def test_exclusions_do_not_become_positive_requests(self):
        for text in ("Konser istemiyorum", "I don't want theatre", "Tiyatro olmasin"):
            self.assertEqual(guard_request(text)[1], "only_exclusions")
            self.assertEqual(parse_request(text), "unclear")

    def test_one_positive_category_survives_other_exclusion(self):
        for text in ("Tiyatro istemiyorum, konser ariyorum", "No theatre please, I want a concert"):
            cleaned, reason = guard_request(text)
            self.assertIsNone(reason)
            self.assertEqual(parse_request(text), "concert")
            self.assertNotIn("tiyatro", cleaned)
            self.assertNotIn("theatre", cleaned)

    def test_music_player_command_is_not_an_attendance_request(self):
        for text in ("Spotify'da bir konser kaydi ac", "Play a concert recording on YouTube", "Play my favorite album", "Muzik acar misin?"):
            self.assertEqual(guard_request(text)[1], "music_player_command")

    def test_shared_exclusion_applies_to_both_categories(self):
        self.assertEqual(guard_request("I do not want theatre or stand-up")[1], "only_exclusions")

    def test_movie_request_is_outside_the_supported_activities(self):
        self.assertEqual(guard_request("Sinemada komedi filmi izlemek istiyorum")[1], "unsupported_activity")

    def test_correction_keeps_remaining_text_after_exclusion(self):
        cleaned, reason = guard_request("Konser olmasin, oyuncularin sahneledigi bir eser olsun")
        self.assertIsNone(reason)
        self.assertNotIn("konser", cleaned)

    def test_player_reference_can_coexist_with_attendance_request(self):
        text = "I heard this band on Spotify and want tickets for a concert"
        self.assertIsNone(guard_request(text)[1])
        self.assertEqual(parse_request(text), "concert")

    def test_parser_cannot_infer_vague_activity(self):
        self.assertEqual(parse_request("Biraz gulelim"), "unclear")
        self.assertEqual(parse_request("Something fun tonight"), "unclear")


if __name__ == "__main__":
    unittest.main()
