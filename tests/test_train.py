import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from classifier import LABELS
from train import read_data, split_data, select_model, comparison_records, train


def fixture_rows():
    return [{"id":str(i*60+j), "text":label.replace("_"," ")+" fixture", "intent":label,
             "scenario":label.split("_")[0], "partition":part}
            for i,part in enumerate(("train","dev","test")) for j,label in enumerate(LABELS)]

class TrainTests(unittest.TestCase):
    def test_official_split_and_tiny_class(self):
        rows=fixture_rows(); parts=split_data(rows)
        self.assertEqual(parts,split_data(rows,99))
        self.assertEqual([len(x) for x in parts.values()],[60]*3)
        self.assertEqual(set(sum(parts.values(),[])),set(range(180)))
        with self.assertRaises(ValueError): split_data(rows[:-1])
    def test_data_validation_preserves_official_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"data.jsonl"
            rows=fixture_rows()
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            self.assertEqual(len(read_data(path,min_rows=1)),180)
            with self.assertRaisesRegex(ValueError,"1000"): read_data(path)
            for key,value in (("id",rows[0]["id"]),("intent","unclear"),("scenario","wrong"),("partition","validation"),("text"," ")):
                bad=[dict(r) for r in rows];bad[1][key]=value
                path.write_text("".join(json.dumps(r)+"\n" for r in bad))
                with self.assertRaises(ValueError): read_data(path,min_rows=1)
    def test_selection_ignores_test(self):
        evaluations={"a":{"dev":{"macro_f1":.8},"test":{"macro_f1":0}}, "b":{"dev":{"macro_f1":.2},"test":{"macro_f1":1}}}
        self.assertEqual(select_model(evaluations),"a")
    def test_comparison_is_test_only_deterministic(self):
        rows=fixture_rows();indices=split_data(rows)["test"]
        records=comparison_records(rows,indices)
        self.assertEqual(records,comparison_records(rows,indices))
        self.assertEqual(len(records),40)
        self.assertEqual(len({r["id"] for r in records}),40)
        self.assertTrue({r["id"] for r in records} <= {rows[i]["id"] for i in indices})
    def test_flow_fits_train_and_predicts_test_after_selection(self):
        events=[]
        class Model:
            def fit(self,x,y): events.append(("fit",len(x)));return self
            def predict(self,x): events.append(("predict",len(x)));return list(LABELS)
        rows=fixture_rows()
        with tempfile.TemporaryDirectory() as tmp, patch("train.read_data",return_value=rows), patch("train.make_models",return_value={"logistic_regression_word":Model()}), patch("sklearn.base.clone",side_effect=lambda m:m), patch("joblib.dump"), patch("train.plot_confusion"):
            # Mock supports the one parameter changed by weighting checks.
            Model.set_params=lambda self,**kw:self
            path=Path(tmp)/"data";path.write_text("fixture")
            result=train(path,Path(tmp)/"model.joblib",tmp,allow_small_prototype=True)
        self.assertEqual(events[:4],[("fit",60),("predict",60)]*2)
        self.assertTrue(result["selection"]["final_refit"])
        self.assertFalse(result["selection"]["test_used_for_selection"])
