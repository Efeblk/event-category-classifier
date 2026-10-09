"""Evaluate NB, LR, SVM and optional paid Jev on the same labeled sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

from classifier import LABELS
from jev import JevClient, PROMPT_VERSION
from method_comparison import METHODS, compare_request, validate_artifacts, validate_input


def _records(dataset: object) -> list[dict[str, Any]]:
    if not isinstance(dataset, dict) or not isinstance(dataset.get("records"), list) or not dataset["records"]:
        raise ValueError("Evaluation data needs a non-empty records list.")
    records = dataset["records"]
    ids = set()
    for row in records:
        if not isinstance(row, dict) or row.get("label") not in LABELS:
            raise ValueError("Each evaluation record needs a supported technique label.")
        identifier = row.get("id")
        if type(identifier) not in (str, int) or not str(identifier).strip() or str(identifier) in ids:
            raise ValueError("Each evaluation record needs a unique non-empty id.")
        ids.add(str(identifier))
        validate_input(row.get("text"), row.get("context", ""))
    return records


def evaluate(
    models_path: Path,
    data_path: Path,
    output_path: Path,
    include_jev: bool = False,
) -> dict[str, Any]:
    if type(include_jev) is not bool:
        raise ValueError("include_jev must be true or false.")
    source = data_path.read_bytes()
    try:
        dataset = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Evaluation data must be valid UTF-8 JSON.") from error
    records = _records(dataset)
    partition = dataset.get("partition", "unspecified")
    if not isinstance(partition, str) or not partition.strip():
        raise ValueError("Evaluation partition must be a non-empty string.")
    # Only load locally generated, trusted joblib files: unpickling executes code.
    artifacts = validate_artifacts(joblib.load(models_path))
    first = artifacts[METHODS[0]]
    if first["provenance"]["evaluation_scope"] == "dev_only" and partition != "dev":
        raise ValueError("Development-only artifacts require a dev sample; final test evaluation is pending.")
    jev = JevClient()
    if include_jev:
        # Validate the complete provider payload of every case before reserving an attempt.
        for row in records:
            jev.prepare_request(row["text"], row.get("context", ""))
        if not jev.configured or jev.remaining_calls() < len(records):
            raise ValueError("Configure Jev and enough remaining attempts for the entire set before --include-jev.")
    output: dict[str, Any] = {
        "schema_version": 1,
        "task": first["task"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": dataset.get("name", "unnamed_comparison_sample"),
        "partition": partition,
        "model_evaluation_scope": first["provenance"]["evaluation_scope"],
        "examples": len(records),
        "provenance": dataset.get("provenance", "Unspecified evaluation provenance."),
        "note": "Same original excerpt and context for every method. Macro F1 uses all 14 labels. "
                f"Abstentions count as errors. Sample partition: {partition}. "
                "A development sample does not measure final-test generalization.",
        "data_sha256": hashlib.sha256(source).hexdigest(),
        "training_data_sha256": first["dataset_sha256"],
        "training_code_sha256": first["code_sha256"],
        "training_seed": first["seed"],
        "context_margin": first["provenance"]["context_margin"],
        "method_models": {method: artifacts[method]["model_name"] for method in METHODS},
        "jev_model": jev.model,
        "jev_prompt_version": PROMPT_VERSION,
        "jev_requested": include_jev,
        "labels": list(LABELS),
        "methods": {},
        "records": [],
        "code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("classifier.py", "jev.py", "method_comparison.py", "compare_methods.py")
        },
    }
    for record in records:
        comparison = compare_request(artifacts, record["text"], jev, include_jev, record.get("context", ""))
        output["records"].append({
            "id": record["id"],
            "text": record["text"],
            "context": record.get("context", ""),
            "actual": record["label"],
            "results": comparison["results"],
        })
    for method in (*METHODS, "jev"):
        predictions = [next(result for result in row["results"] if result["method"] == method)
                       for row in output["records"]]
        completed = sum(row["status"] == "ok" for row in predictions)
        result = {
            "completed": completed,
            "total": len(records),
            "failed": sum(row["status"] in ("error", "budget_exhausted") for row in predictions),
            "not_run": sum(row["status"] not in ("ok", "error", "budget_exhausted") for row in predictions),
            "accuracy": None,
            "macro_f1": None,
            "evaluation_status": "evaluated" if completed == len(records) else "not evaluated",
        }
        # A partial provider run must never masquerade as a comparable full score.
        if completed == len(records):
            actual = [row["label"] for row in records]
            labels = [row["label"] for row in predictions]
            result.update(
                accuracy=float(accuracy_score(actual, labels)),
                macro_f1=float(f1_score(actual, labels, labels=list(LABELS), average="macro", zero_division=0)),
                per_class=classification_report(actual, labels, labels=list(LABELS), output_dict=True, zero_division=0),
            )
        output["methods"][method] = result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    for method, result in output["methods"].items():
        score = f"{result['accuracy']:.1%}" if result["accuracy"] is not None else "not evaluated"
        print(f"{method}: {score}; {result['completed']}/{result['total']} completed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("artifacts/method_models.joblib"))
    parser.add_argument("--data", type=Path, default=Path("reports/comparison_set.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/method_comparison.json"))
    parser.add_argument("--include-jev", action="store_true", help="Make one paid attempt per example. Requires explicit authorization; no retries.")
    args = parser.parse_args()
    try:
        evaluate(args.models, args.data, args.output, args.include_jev)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
