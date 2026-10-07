"""Evaluate the three methods on exactly the same labeled texts."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

from classifier import OUTPUT_LABELS
from jev import JevClient, PROMPT_VERSION
from method_comparison import compare_request
from parser import parser_scores


def evaluate(model_path, data_path, output_path, include_jev=False):
    source = data_path.read_bytes()
    dataset = json.loads(source)
    records = dataset.get("records")
    if not isinstance(records, list) or not records or any(
        not isinstance(row, dict) or row.get("label") not in OUTPUT_LABELS
        or not isinstance(row.get("text"), str) or not row["text"].strip()
        or len(row["text"]) > (1000 if include_jev else 5000) for row in records
    ):
        raise ValueError("Each evaluation record needs valid text and a supported label.")
    jev = JevClient()
    if include_jev and (not jev.configured or jev.remaining_calls() < len(records)):
        raise ValueError("Configure Jev and enough remaining calls for the whole set before --include-jev.")
    artifact = joblib.load(model_path)
    output = {
        "schema_version": 1, "task": artifact["task"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": dataset.get("name", "unnamed_set"), "examples": len(records),
        "provenance": dataset.get("provenance", "Unspecified evaluation provenance."),
        "note": "Same inputs for every method. Abstaining with unclear counts as wrong unless unclear is the label.",
        "data_sha256": hashlib.sha256(source).hexdigest(),
        "training_data_sha256": artifact["dataset_sha256"],
        "logistic_regression_model": artifact["model_name"],
        "jev_model": jev.model, "jev_prompt_version": PROMPT_VERSION,
        "jev_requested": include_jev, "methods": {}, "records": [],
        "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("classifier.py", "parser.py", "jev.py", "method_comparison.py", "compare_methods.py")},
    }
    # Score only the labels that occur in this set, so an unused abstain label adds no zero F1.
    labels_in_set = [label for label in OUTPUT_LABELS if any(row["label"] == label for row in records)]
    for record in records:
        comparison = compare_request(artifact, record["text"], jev, include_jev)
        output["records"].append({"text": record["text"], "actual": record["label"],
                                  "results": comparison["results"]})
    for method in ("parser", "logistic_regression", "jev"):
        predictions = [next(result for result in row["results"] if result["method"] == method)
                       for row in output["records"]]
        completed = [index for index, result in enumerate(predictions) if result["status"] == "ok"]
        result = {"completed": len(completed), "total": len(records),
                  "failed": sum(row["status"] in ("error", "budget_exhausted") for row in predictions),
                  "not_run": sum(row["status"] not in ("ok", "error", "budget_exhausted") for row in predictions),
                  "accuracy": None, "macro_f1": None}
        # Do not present a partial provider run as a complete comparable score.
        if len(completed) == len(records):
            actual = [row["label"] for row in records]
            labels = [row["label"] for row in predictions]
            result.update(accuracy=accuracy_score(actual, labels),
                          macro_f1=f1_score(actual, labels, labels=labels_in_set, average="macro", zero_division=0),
                          per_class=classification_report(actual, labels, labels=labels_in_set,
                                                          output_dict=True, zero_division=0))
        if method == "parser" and len(completed) == len(records):
            result.update(parser_scores(actual, labels))
        result["evaluation_status"] = "evaluated" if len(completed) == len(records) else "not evaluated"
        output["methods"][method] = result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    for method, result in output["methods"].items():
        score = f"{result['accuracy']:.1%}" if result["accuracy"] is not None else "not evaluated"
        print(f"{method}: {score}; {result['completed']}/{result['total']} completed")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/lr_model.joblib"))
    parser.add_argument("--data", type=Path, default=Path("reports/comparison_set.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/method_comparison.json"))
    parser.add_argument("--include-jev", action="store_true", help="Make one paid API attempt per example. No automatic retries.")
    args = parser.parse_args()
    try:
        evaluate(args.model, args.data, args.output, args.include_jev)
    except ValueError as error:
        parser.error(str(error))
