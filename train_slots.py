"""Train the removable slot extension; intent training does not import this file."""
import argparse
import csv
import hashlib
from pathlib import Path
import joblib
from sklearn.feature_extraction import DictVectorizer
from slots import HIGHLIGHT_SLOTS, PARSER_SLOT_TYPES, exact_match, make_slot_models, parser_tags, predict_tags, slot_scores, token_features
from classifier import _validated_model
from train import read_data, split_data, write_json, BENCHMARK


def train_slots(data_path="data/massive_tr.jsonl", intent_path="artifacts/model.joblib", model_path="artifacts/slot_model.joblib", report_dir="reports", seed=42):
    rows = read_data(data_path)
    parts = split_data(rows)
    sentences = {p:[rows[i]["tokens"] for i in indices] for p,indices in parts.items()}
    truth = {p:[rows[i]["tags"] for i in indices] for p,indices in parts.items()}
    for row in rows:
        if row["tokens"] != row["text"].split() or len(row["tokens"]) != len(row["tags"]):
            raise ValueError("Invalid token/tag alignment.")
    features = [token_features(tokens,i) for tokens in sentences["train"] for i in range(len(tokens))]
    vectorizer = DictVectorizer()
    x = vectorizer.fit_transform(features)
    y = [tag for tags in truth["train"] for tag in tags]
    tag_labels = sorted(set(y))
    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    artifact = {"vectorizer":vectorizer,"tags":tag_labels,"task":"slot_token_classification",
                "dataset_domain":"massive_tr","dataset_sha256":digest,"seed":seed}
    intent_artifact = joblib.load(intent_path)
    if intent_artifact.get("dataset_sha256") != digest: raise ValueError("Intent and slot data hashes differ.")
    intent_model = _validated_model(intent_artifact)
    # Intent winner is frozen by train.py; no intent selection in the extension.
    intent_predictions = {p:list(map(str,intent_model.predict([rows[i]["text"] for i in parts[p]]))) for p in ("dev","test")}
    actual_intents = {p:[rows[i]["intent"] for i in parts[p]] for p in ("dev","test")}
    models = make_slot_models(seed)
    evaluations = {}
    for name,model in models.items():
        print(f"fit slots {name}",flush=True)
        if model is not None: model.fit(x,y)
        predictions = predict_tags(artifact | {"model":model},sentences["dev"])
        evaluations[name] = {"dev":slot_scores(truth["dev"],predictions)}
        evaluations[name]["dev"]["exact_match"] = exact_match(actual_intents["dev"],intent_predictions["dev"],truth["dev"],predictions)
    selected = max(models,key=lambda name:(evaluations[name]["dev"]["f1"],name))
    test_predictions = {}
    for name,model in models.items():
        predictions = predict_tags(artifact | {"model":model},sentences["test"])
        test_predictions[name] = predictions
        evaluations[name]["test"] = slot_scores(truth["test"],predictions)
        evaluations[name]["test"]["exact_match"] = exact_match(actual_intents["test"],intent_predictions["test"],truth["test"],predictions)
    destination = Path(model_path);destination.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(artifact | {"model":models[selected],"model_name":selected},destination)
    parser_metrics = {p:slot_scores(truth[p],[parser_tags(tokens) for tokens in sentences[p]],PARSER_SLOT_TYPES) for p in ("dev","test")}
    metadata = {"seed":seed,"dataset_sha256":digest,"selected_model":selected,"selection_metric":"dev_span_micro_f1",
                "test_used_for_selection":False,"intent_model":intent_artifact["model_name"],
                "benchmark":BENCHMARK,"models":evaluations,"parser":{"scope":"date, time, timeofday only",**parser_metrics},
                "highlight_slots":{k:evaluations[selected]["test"]["per_slot"].get(k,{}) for k in HIGHLIGHT_SLOTS},
                "decoding":"I-x after O or a different type becomes B-x; exact type and token boundaries required.",
                "jev_slots":"Not implemented: API has Choice, Noul and Score, no arbitrary span extraction primitive.",
                "annotation_note":"Whitespace suffixes are kept. Similar time expressions can have inconsistent span labels; official labels are preserved."}
    directory = Path(report_dir);write_json(directory/"slot_metrics.json",metadata)
    with (directory/"slot_errors.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["id","text","gold_tags","predicted_tags","boundary_errors","type_confusions"])
        for i,gold,predicted in zip(parts["test"],truth["test"],test_predictions[selected]):
            if gold != predicted:
                errors=slot_scores([gold],[predicted])["errors"]
                writer.writerow([rows[i]["id"],rows[i]["text"]," ".join(gold)," ".join(predicted),errors["boundary"],errors["type_confusion"]])
    return metadata


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default="data/massive_tr.jsonl")
    parser.add_argument("--intent-model",default="artifacts/model.joblib")
    parser.add_argument("--model",default="artifacts/slot_model.joblib")
    parser.add_argument("--reports-dir",default="reports")
    args=parser.parse_args()
    try: print(train_slots(args.data,args.intent_model,args.model,args.reports_dir)["selected_model"])
    except ValueError as error: parser.error(str(error))
