"""Evaluate the frozen prototype on separately authored challenge examples."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

from classifier import LABELS, classify_request, normalize_text
from request_policy import classify_input, keyword_baseline


def evaluate(model_path, challenge_path, output_path):
    raw = challenge_path.read_bytes()
    challenge = json.loads(raw)
    records = challenge["records"]
    if not records or any(record["label"] not in LABELS for record in records):
        raise ValueError("Challenge records must use the four request labels.")
    # Reject direct overlap with the prepared dataset. This is not a proof of semantic independence.
    training_path = Path("data/requests.csv")
    if training_path.exists():
        with training_path.open(encoding="utf-8", newline="") as handle:
            known = {normalize_text(row["text"]) for row in csv.DictReader(handle)}
        if any(normalize_text(record["text"]) in known for record in records):
            raise ValueError("Challenge text overlaps the prepared dataset.")
    artifact = joblib.load(model_path)
    actual = [record["label"] for record in records]
    modes = {
        "model_with_vocabulary_guard": lambda text: classify_request(artifact, text),
        "demo_with_input_guards": lambda text: classify_input(artifact, text),
        "keyword_baseline": lambda text: {"label": keyword_baseline(text)},
    }
    output = {
        "task": "user_request_classification", "examples": len(records),
        "provenance": challenge["provenance"], "real_user_accuracy_established": False,
        "challenge_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_sha256": artifact["dataset_sha256"], "model": artifact["model_name"],
        "models": {},
    }
    for name, predict in modes.items():
        predictions = [predict(record["text"]) for record in records]
        labels = [result["label"] for result in predictions]
        output["models"][name] = {
            "accuracy": accuracy_score(actual, labels),
            "macro_f1": f1_score(actual, labels, labels=list(LABELS), average="macro"),
            "per_class": classification_report(actual, labels, labels=list(LABELS), output_dict=True, zero_division=0),
            "records": [{"text": record["text"], "actual": record["label"], **prediction}
                        for record, prediction in zip(records, predictions)],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    for name, result in output["models"].items():
        print(f"{name}: {result['accuracy']:.1%} synthetic challenge accuracy")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--challenge", type=Path, default=Path("data/request_challenge.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/request_challenge.json"))
    args = parser.parse_args()
    evaluate(args.model, args.challenge, args.output)
