"""Train sparse intent models on MASSIVE Turkish official partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from classifier import ARTIFACT_TASK, DATASET_DOMAIN, LABELS, make_models, normalize_text
from parser import parse_request, parser_scores


MIN_ROWS = 1000
PARTITIONS = ("train", "dev", "test")
BENCHMARK = {
    "source": "https://arxiv.org/html/2204.08582v2#A4",
    "tables": "Appendix D, Tables 7–9; quoted, not reproduced",
    "training": "Paper: all 51 locales; ours: tr-TR train only. Same official tr-TR test; 60 output intents (59 represented in test).",
    "models": {
        "XLM-R base": {
            "intent_accuracy": .863, "intent_accuracy_pm": .012,
            "slot_f1": .749, "slot_f1_pm": .007,
            "exact_match": .652, "exact_match_pm": .017,
        },
        "mT5 encoder-only": {
            "intent_accuracy": .871, "intent_accuracy_pm": .012,
            "slot_f1": .761, "slot_f1_pm": .007,
            "exact_match": .677, "exact_match_pm": .017,
        },
        "mT5 text-to-text": {
            "intent_accuracy": .861, "intent_accuracy_pm": .012,
            "slot_f1": .779, "slot_f1_pm": .006,
            "exact_match": .681, "exact_match_pm": .017,
        },
    },
}


def read_data(path: str | Path, min_rows: int = MIN_ROWS) -> list[dict]:
    """Read prepared rows and check the fields that training relies on."""
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1.")
    if not Path(path).is_file():
        raise ValueError(f"Data file does not exist: {path}")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if len(rows) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows. Found {len(rows)}.")

    ids = set()
    for row in rows:
        if not {"id", "text", "intent", "scenario", "partition"}.issubset(row):
            raise ValueError("Missing prepared data fields.")
        row_id = str(row["id"])
        if not row_id.strip() or row_id in ids:
            raise ValueError("Empty or duplicate row id.")
        ids.add(row_id)
        if not isinstance(row["text"], str) or not normalize_text(row["text"]):
            raise ValueError("Every row needs non-empty text.")
        if row["intent"] not in LABELS or row["scenario"] != row["intent"].split("_")[0]:
            raise ValueError("Unsupported intent or inconsistent scenario.")
        if row["partition"] not in PARTITIONS:
            raise ValueError("Unsupported official partition.")
    if {row["intent"] for row in rows} != set(LABELS):
        raise ValueError("Dataset must contain all 60 intents.")
    # Duplicates, including contradictory labels, are official data: audit them, never drop.
    return rows


def split_data(rows: list[dict], seed: int = 42) -> dict[str, list[int]]:
    """Return row indices of the official partitions. The seed is unused."""
    parts = {part: [i for i, row in enumerate(rows) if row["partition"] == part] for part in PARTITIONS}
    if sum(map(len, parts.values())) != len(rows):
        raise ValueError("Unsupported official partition.")
    for part, indices in parts.items():
        if not indices:
            raise ValueError(f"The {part} split is empty.")
    if {rows[i]["intent"] for i in parts["train"]} != set(LABELS):
        raise ValueError("The train split must contain all intents.")
    # No per-class minimum: cooking_query has only four train examples.
    return parts


def scores(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    labels = list(LABELS)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "per_intent": classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0),
    }


def select_model(evaluations: dict[str, Any], names: list[str] | None = None) -> str:
    """Pick the highest dev macro F1. Test scores are never read."""
    candidates = names or list(evaluations)
    return max(candidates, key=lambda name: (evaluations[name]["dev"]["macro_f1"], name))


def comparison_records(rows: list[dict], test_indices: list[int], seed: int = 42) -> list[dict]:
    """Sample 40 test requests for the same-input method comparison."""
    generator = random.Random(seed)
    indices = sorted(test_indices, key=lambda i: int(rows[i]["id"]))
    sample = generator.sample(indices, min(40, len(indices)))
    return [{"id": rows[i]["id"], "text": rows[i]["text"], "label": rows[i]["intent"]} for i in sample]


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train(
    data_path: str | Path = "data/massive_tr.jsonl",
    model_path: str | Path = "artifacts/model.joblib",
    report_dir: str | Path = "reports",
    seed: int = 42,
    allow_small_prototype: bool = False,
) -> dict[str, Any]:
    import joblib
    from sklearn.base import clone

    rows = read_data(data_path, min_rows=1 if allow_small_prototype else MIN_ROWS)
    parts = split_data(rows)
    texts = {part: [rows[i]["text"] for i in indices] for part, indices in parts.items()}
    labels = {part: [rows[i]["intent"] for i in indices] for part, indices in parts.items()}

    models = make_models(seed)
    # Predeclared dev check: each LR and SVM candidate also runs without class weights.
    weighted = [name for name in models if name.startswith(("logistic_regression", "linear_svm"))]
    for name in weighted:
        models[name + "_unbalanced"] = clone(models[name]).set_params(classifier__class_weight=None)

    # Every model is fitted on train only and compared on dev.
    evaluations = {}
    for name, model in models.items():
        print(f"fit {name}", flush=True)
        model.fit(texts["train"], labels["train"])
        evaluations[name] = {"dev": scores(labels["dev"], model.predict(texts["dev"]))}
    selected = select_model(evaluations)
    lr_selected = select_model(evaluations, [name for name in models if name.startswith("logistic_regression_")])

    # Test is predicted only after both winners are chosen.
    predictions = {}
    for name, model in models.items():
        predictions[name] = [str(label) for label in model.predict(texts["test"])]
        evaluations[name]["test"] = scores(labels["test"], predictions[name])

    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name, path in ((selected, destination), (lr_selected, destination.with_name("lr_model.joblib"))):
        joblib.dump({
            "model": models[name],
            "model_name": name,
            "labels": LABELS,
            "dataset_sha256": digest,
            "seed": seed,
            "task": ARTIFACT_TASK,
            "dataset_domain": DATASET_DOMAIN,
            "provenance": "MASSIVE v1.0 tr-TR official train",
        }, path)

    metadata = {
        "seed": seed,
        "task": ARTIFACT_TASK,
        "dataset_domain": DATASET_DOMAIN,
        "dataset": {"sha256": digest, "rows": len(rows)},
        "split": {
            part: {
                "rows": len(indices),
                "intents": dict(Counter(labels[part])),
                "missing_intents": sorted(set(LABELS) - set(labels[part])),
            }
            for part, indices in parts.items()
        },
        "duplicate_audit": duplicate_audit(rows),
        "selection": {
            "metric": "dev_macro_f1",
            "selected_model": selected,
            "logistic_regression_model": lr_selected,
            "fit_split": "train",
            "test_used_for_selection": False,
        },
        "class_weight_check": class_weight_check(evaluations, weighted),
        "benchmark": BENCHMARK,
        "models": evaluations,
        "parser": {
            part: parser_scores(labels[part], [parse_request(text) for text in texts[part]])
            for part in ("dev", "test")
        },
        "top_confused_intent_pairs": top_confusions(labels["test"], predictions[selected]),
    }

    directory = Path(report_dir)
    write_json(directory / "metrics.json", metadata)
    write_json(directory / "comparison_set.json", {
        "name": "40 MASSIVE tr-TR official test requests",
        "provenance": "Uniform sample without replacement, seed 42. Not the full benchmark.",
        "license": "CC BY 4.0",
        "records": comparison_records(rows, parts["test"], seed),
    })
    with (directory / "errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", "gold", "predicted"])
        for i, predicted in zip(parts["test"], predictions[selected]):
            if rows[i]["intent"] != predicted:
                writer.writerow([rows[i]["id"], rows[i]["text"], rows[i]["intent"], predicted])
    plot_confusion(labels["test"], predictions[selected], directory / "confusion_matrix.png")
    return metadata


def duplicate_audit(rows: list[dict]) -> dict[str, Any]:
    """Count normalized texts that cross partitions or carry more than one intent."""
    partitions, intents = defaultdict(set), defaultdict(set)
    for row in rows:
        text = normalize_text(row["text"])
        partitions[text].add(row["partition"])
        intents[text].add(row["intent"])
    return {
        "cross_partition_texts": sum(len(found) > 1 for found in partitions.values()),
        "mixed_intent_texts": sum(len(found) > 1 for found in intents.values()),
        "action": "preserved official rows",
    }


def class_weight_check(evaluations: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Compare balanced and unbalanced class weights on dev."""
    result = {}
    for name in names:
        balanced = evaluations[name]["dev"]["macro_f1"]
        unbalanced = evaluations[name + "_unbalanced"]["dev"]["macro_f1"]
        result[name] = {
            "balanced_dev_f1": balanced,
            "unbalanced_dev_f1": unbalanced,
            "preferred": "balanced" if balanced >= unbalanced else "unbalanced",
        }
    return result


def top_confusions(actual: list[str], predicted: list[str], count: int = 15) -> list[dict]:
    pairs = Counter((gold, guess) for gold, guess in zip(actual, predicted) if gold != guess)
    return [{"gold": gold, "predicted": guess, "count": n} for (gold, guess), n in pairs.most_common(count)]


def plot_confusion(actual: list[str], predicted: list[str], path: Path) -> None:
    """Plot an 18x18 scenario confusion matrix; 60x60 intents would be unreadable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    def scenario(label: str) -> str:
        return label.split("_")[0]

    scenarios = sorted({scenario(label) for label in LABELS})
    matrix = confusion_matrix(
        [scenario(label) for label in actual],
        [scenario(label) for label in predicted],
        labels=scenarios,
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(matrix, display_labels=scenarios).plot(
        ax=ax, cmap="Blues", xticks_rotation=90, values_format="d", colorbar=False
    )
    ax.set_title("Official test: scenario confusion (selected intent model)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/massive_tr.jsonl")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-small-prototype", action="store_true")
    args = parser.parse_args()
    try:
        result = train(args.data, args.model, args.reports_dir, args.seed, args.allow_small_prototype)
    except ValueError as error:
        parser.error(str(error))
    print(result["selection"])


if __name__ == "__main__":
    main()
