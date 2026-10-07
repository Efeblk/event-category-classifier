import unittest
from prepare_massive import parse_annotation
from slots import repair_bio, spans, slot_scores, token_features, parser_tags, exact_match, display_spans, predict_tags

class SlotTests(unittest.TestCase):
    def test_bio_alignment(self):
        tokens,tags=parse_annotation("yarın saat beşte gel", "[date : yarın] saat [time : beşte] gel")
        self.assertEqual(len(tokens),len(tags))
        self.assertEqual(spans(tags),{("date",0,1),("time",2,3)})
    def test_repair_and_adjacent_spans(self):
        self.assertEqual(repair_bio(["I-date","I-date","I-time","O","I-time"]),["B-date","I-date","B-time","O","B-time"])
        self.assertEqual(spans(["B-date","B-date"]),{("date",0,1),("date",1,2)})
        with self.assertRaises(ValueError): repair_bio(["X-date"])
    def test_exact_span_f1_and_error_types(self):
        actual=[["B-date","I-date","B-time"]]
        predicted=[["B-date","O","B-date"]]
        result=slot_scores(actual,predicted)
        self.assertEqual(result["f1"],0)
        self.assertEqual(result["errors"],{"boundary":1,"type_confusion":1,"missed":0,"spurious":0})
        self.assertEqual(slot_scores(actual,actual)["f1"],1)
        self.assertEqual(slot_scores([["O"]],[["O"]])["f1"],0)
        result=slot_scores([["B-date","B-time"]],[["B-date","O"]])
        self.assertEqual(result["precision"],1);self.assertEqual(result["recall"],.5);self.assertAlmostEqual(result["f1"],2/3)
    def test_parser_scope_and_exact_match(self):
        self.assertEqual(slot_scores([["B-date","B-person"]],[["B-date","O"]],["date"])["f1"],1)
        self.assertEqual(exact_match(["a"],["b"],[["O"]],[["O"]]),0)
        self.assertEqual(exact_match(["a"],["a"],[["B-date"]],[["I-date"]]),1)
        for text in ("yarın akşam dokuzda gel", "bugün saat 5'te gel"):
            tags=parser_tags(text.split());self.assertIn("B-date",tags);self.assertIn("B-time",tags)
    def test_features_and_offsets(self):
        features=token_features(["İstanbul'da","5","gel"],0)
        self.assertEqual(features["word"],"istanbul'da")
        self.assertEqual(features["prefix3"],"ist");self.assertEqual(features["suffix2"],"da")
        self.assertTrue(features["start"]);self.assertTrue(features["capital"]);self.assertTrue(features["apostrophe"])
        self.assertEqual(features["word+2"],"gel");self.assertEqual(features["word-1"],"<BOUNDARY>")
        self.assertTrue(token_features(["5"],0)["number"])
        self.assertEqual(display_spans("  yarın  gel",["B-date","O"]),[{"type":"date","start":2,"end":7,"text":"yarın"}])
    def test_slot_artifact_validation(self):
        with self.assertRaises(ValueError): predict_tags({},[["hi"]])
