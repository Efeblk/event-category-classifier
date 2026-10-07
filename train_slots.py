"""Train the removable slot extension; intent training does not import this file."""

import argparse
import csv
import hashlib
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer

from classifier import _validated_model
from slots import (
    HIGHLIGHT_SLOTS,
    PARSER_SLOT_TYPES,
    compact_indices,
    exact_match,
    make_slot_models,
    parser_tags,
    predict_tags,
    slot_scores,
    token_features,
)
from train import BENCHMARK, read_data, split_data, write_json


def train_slots(
    data_path="data/massive_tr.jsonl",
    intent_path="artifacts/model.joblib",
    model_path="artifacts/slot_model.joblib",
    report_dir="reports",
    seed=42,
):
    rows = read_data(data_path)
    parts = split_data(rows)
    for row in rows:
        if row["tokens"] != row["text"].split() or len(row["tokens"]) != len(row["tags"]):
            raise ValueError("Invalid token/tag alignment.")
    sentences = {part: [rows[i]["tokens"] for i in indices] for part, indices in parts.items()}
    truth = {part: [rows[i]["tags"] for i in indices] for part, indices in parts.items()}

    # One training example per token.
    features = [token_features(tokens, i) for tokens in sentences["train"] for i in range(len(tokens))]
    vectorizer = DictVectorizer()
    x = compact_indices(vectorizer.fit_transform(features))
    y = [tag for tags in truth["train"] for tag in tags]
    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    artifact = {
        "vectorizer": vectorizer,
        "tags": sorted(set(y)),
        "task": "slot_token_classification",
        "dataset_domain": "massive_tr",
        "dataset_sha256": digest,
        "seed": seed,
    }

    # The intent winner is frozen by train.py; it is only used for joint exact match.
    intent_artifact = joblib.load(intent_path)
    if intent_artifact.get("dataset_sha256") != digest:
        raise ValueError("Intent and slot data hashes differ.")
    intent_model = _validated_model(intent_artifact)
    actual_intents, intent_predictions = {}, {}
    for part in ("dev", "test"):
        actual_intents[part] = [rows[i]["intent"] for i in parts[part]]
        intent_predictions[part] = [str(label) for label in intent_model.predict([rows[i]["text"] for i in parts[part]])]

    def evaluate(model, part):
        predictions = predict_tags(artifact | {"model": model}, sentences[part])
        result = slot_scores(truth[part], predictions)
        result["exact_match"] = exact_match(actual_intents[part], intent_predictions[part], truth[part], predictions)
        return result, predictions

    models = make_slot_models(seed)
    evaluations = {}
    for name, model in models.items():
        print(f"fit slots {name}", flush=True)
        if model is not None:
            model.fit(x, y)
        evaluations[name] = {"dev": evaluate(model, "dev")[0]}
    selected = max(models, key=lambda name: (evaluations[name]["dev"]["f1"], name))

    # Test is scored only after the dev-based selection.
    test_predictions = {}
    for name, model in models.items():
        evaluations[name]["test"], test_predictions[name] = evaluate(model, "test")

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact | {"model": models[selected], "model_name": selected}, destination)

    parser_metrics = {
        part: slot_scores(truth[part], [parser_tags(tokens) for tokens in sentences[part]], PARSER_SLOT_TYPES)
        for part in ("dev", "test")
    }
    metadata = {
        "seed": seed,
        "dataset_sha256": digest,
        "selected_model": selected,
        "selection_metric": "dev_span_micro_f1",
        "test_used_for_selection": False,
        "intent_model": intent_artifact["model_name"],
        "benchmark": BENCHMARK,
        "models": evaluations,
        "parser": {"scope": "date, time, timeofday only", **parser_metrics},
        "highlight_slots": {kind: evaluations[selected]["test"]["per_slot"].get(kind, {}) for kind in HIGHLIGHT_SLOTS},
        "decoding": "I-x after O or a different type becomes B-x; exact type and token boundaries required.",
        "jev_slots": "Not implemented: API has Choice, Noul and Score, no arbitrary span extraction primitive.",
        "annotation_note": "Whitespace suffixes are kept. Similar time expressions can have inconsistent span labels; official labels are preserved.",
    }
    directory = Path(report_dir)
    write_json(directory / "slot_metrics.json", metadata)
    with (directory / "slot_errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", "gold_tags", "predicted_tags", "boundary_errors", "type_confusions"])
        for i, gold, predicted in zip(parts["test"], truth["test"], test_predictions[selected]):
            if gold != predicted:
                errors = slot_scores([gold], [predicted])["errors"]
                writer.writerow([
                    rows[i]["id"], rows[i]["text"], " ".join(gold), " ".join(predicted),
                    errors["boundary"], errors["type_confusion"],
                ])
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/massive_tr.jsonl")
    parser.add_argument("--intent-model", default="artifacts/model.joblib")
    parser.add_argument("--model", default="artifacts/slot_model.joblib")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    try:
        print(train_slots(args.data, args.intent_model, args.model, args.reports_dir)["selected_model"])
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
