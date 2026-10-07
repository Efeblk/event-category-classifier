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
BENCHMARK = {
    "source": "https://arxiv.org/html/2204.08582v2#A4",
    "tables": "Appendix D, Tables 7–9; quoted, not reproduced",
    "training": "Paper: all 51 locales; ours: tr-TR train only. Same official tr-TR test; 60 output intents (59 represented in test).",
    "models": {
        "XLM-R base": {"intent_accuracy": .863, "intent_accuracy_pm": .012, "slot_f1": .749, "slot_f1_pm": .007, "exact_match": .652, "exact_match_pm": .017},
        "mT5 encoder-only": {"intent_accuracy": .871, "intent_accuracy_pm": .012, "slot_f1": .761, "slot_f1_pm": .007, "exact_match": .677, "exact_match_pm": .017},
        "mT5 text-to-text": {"intent_accuracy": .861, "intent_accuracy_pm": .012, "slot_f1": .779, "slot_f1_pm": .006, "exact_match": .681, "exact_match_pm": .017},
    },
}


def read_data(path: str | Path, min_rows: int = MIN_ROWS) -> list[dict]:
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1.")
    if not Path(path).is_file():
        raise ValueError(f"Data file does not exist: {path}")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if len(rows) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows. Found {len(rows)}.")
    ids = set()
    duplicates = defaultdict(set)
    for row in rows:
        if not {"id", "text", "intent", "scenario", "partition"}.issubset(row):
            raise ValueError("Missing prepared data fields.")
        if not str(row["id"]).strip() or str(row["id"]) in ids:
            raise ValueError("Empty or duplicate row id.")
        ids.add(str(row["id"]))
        if not isinstance(row["text"], str) or not normalize_text(row["text"]):
            raise ValueError("Every row needs non-empty text.")
        if row["intent"] not in LABELS or row["scenario"] != row["intent"].split("_")[0]:
            raise ValueError("Unsupported intent or inconsistent scenario.")
        if row["partition"] not in {"train", "dev", "test"}:
            raise ValueError("Unsupported official partition.")
        duplicates[normalize_text(row["text"])].add(row["partition"])
    if {row["intent"] for row in rows} != set(LABELS):
        raise ValueError("Dataset must contain all 60 intents.")
    # Duplicates, including contradictory labels, are official data; audit, never drop.
    return rows


def split_data(rows: list[dict], seed: int = 42) -> dict[str, list[int]]:
    parts = {part: [i for i, row in enumerate(rows) if row["partition"] == part]
             for part in ("train", "dev", "test")}
    if sum(map(len, parts.values())) != len(rows):
        raise ValueError("Unsupported official partition.")
    for part, indices in parts.items():
        if not indices:
            raise ValueError(f"The {part} split is empty.")
        if part == "train" and {rows[i]["intent"] for i in indices} != set(LABELS):
            raise ValueError("The train split must contain all intents.")
    # No five-example minimum: cooking_query has only four train examples.
    return parts


def scores(actual, predicted):
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=list(LABELS), average="macro", zero_division=0)),
            "per_intent": classification_report(actual, predicted, labels=list(LABELS), output_dict=True, zero_division=0)}


def select_model(evaluations, names=None):
    return max(names or evaluations, key=lambda name: (evaluations[name]["dev"]["macro_f1"], name))


def comparison_records(rows, test_indices, seed=42):
    generator = random.Random(seed)
    indices = sorted(test_indices, key=lambda i: int(rows[i]["id"]))
    return [{"id": rows[i]["id"], "text": rows[i]["text"], "label": rows[i]["intent"]}
            for i in generator.sample(indices, min(40, len(indices)))]


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def train(data_path="data/massive_tr.jsonl", model_path="artifacts/model.joblib", report_dir="reports", seed=42, allow_small_prototype=False):
    import joblib
    from sklearn.base import clone
    rows = read_data(data_path, min_rows=1 if allow_small_prototype else MIN_ROWS)
    parts = split_data(rows)
    texts = {p: [rows[i]["text"] for i in indices] for p, indices in parts.items()}
    labels = {p: [rows[i]["intent"] for i in indices] for p, indices in parts.items()}
    models = make_models(seed)
    # Four predeclared dev checks of weighting, with no test-driven tuning.
    for name, model in list(models.items()):
        if name.startswith(("logistic_regression", "linear_svm")):
            models[name+"_unbalanced"] = clone(model).set_params(classifier__class_weight=None)
    evaluations = {}
    for name, model in models.items():
        print(f"fit {name}", flush=True)
        model.fit(texts["train"], labels["train"])
        evaluations[name] = {"dev": scores(labels["dev"], model.predict(texts["dev"]))}
    selected = select_model(evaluations)
    lr_selected = select_model(evaluations, [n for n in models if n.startswith("logistic_regression_")])
    # Both winners are chosen before any test prediction; refit on train alone.
    for name in {selected, lr_selected}:
        models[name] = clone(models[name]).fit(texts["train"], labels["train"])
    predictions = {}
    for name, model in models.items():
        predictions[name] = list(map(str, model.predict(texts["test"])))
        evaluations[name]["test"] = scores(labels["test"], predictions[name])
    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name, path in ((selected, destination), (lr_selected, destination.with_name("lr_model.joblib"))):
        joblib.dump({"model": models[name], "model_name": name, "labels": LABELS,
                     "dataset_sha256": digest, "seed": seed, "task": ARTIFACT_TASK,
                     "dataset_domain": DATASET_DOMAIN, "provenance": "MASSIVE v1.0 tr-TR official train"}, path)
    duplicate_parts = defaultdict(set)
    duplicate_labels = defaultdict(set)
    for row in rows:
        normalized = normalize_text(row["text"])
        duplicate_parts[normalized].add(row["partition"])
        duplicate_labels[normalized].add(row["intent"])
    confused = Counter((a,b) for a,b in zip(labels["test"],predictions[selected]) if a != b)
    metadata = {"seed": seed, "task": ARTIFACT_TASK, "dataset_domain": DATASET_DOMAIN,
                "dataset": {"sha256": digest, "rows": len(rows)},
                "split": {p: {"rows": len(indices), "intents": dict(Counter(labels[p])), "missing_intents": sorted(set(LABELS)-set(labels[p]))} for p,indices in parts.items()},
                "duplicate_audit": {"cross_partition_texts": sum(len(p)>1 for p in duplicate_parts.values()),
                                    "mixed_intent_texts": sum(len(p)>1 for p in duplicate_labels.values()), "action": "preserved official rows"},
                "selection": {"metric": "dev_macro_f1", "selected_model": selected, "logistic_regression_model": lr_selected,
                              "fit_split": "train", "final_refit": True, "test_used_for_selection": False},
                "class_weight_check": {name: {"balanced_dev_f1": evaluations[name]["dev"]["macro_f1"],
                                               "unbalanced_dev_f1": evaluations[name+"_unbalanced"]["dev"]["macro_f1"],
                                               "preferred": "balanced" if evaluations[name]["dev"]["macro_f1"] >= evaluations[name+"_unbalanced"]["dev"]["macro_f1"] else "unbalanced"}
                                       for name in models if name.startswith(("logistic_regression", "linear_svm")) and not name.endswith("_unbalanced")},
                "benchmark": BENCHMARK, "models": evaluations,
                "parser": {p: parser_scores(labels[p], [parse_request(text) for text in texts[p]]) for p in ("dev", "test")},
                "top_confused_intent_pairs": [{"gold": a, "predicted": b, "count": n} for (a,b),n in confused.most_common(15)]}
    directory = Path(report_dir)
    write_json(directory/"metrics.json", metadata)
    write_json(directory/"comparison_set.json", {"name": "40 MASSIVE tr-TR official test requests",
               "provenance": "Uniform sample without replacement, seed 42. Not the full benchmark.", "license": "CC BY 4.0",
               "records": comparison_records(rows, parts["test"], seed)})
    with (directory/"errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["id", "text", "gold", "predicted"])
        for i, predicted in zip(parts["test"], predictions[selected]):
            if rows[i]["intent"] != predicted: writer.writerow([rows[i]["id"], rows[i]["text"], rows[i]["intent"], predicted])
    plot_confusion(labels["test"], predictions[selected], directory/"confusion_matrix.png")
    return metadata


def plot_confusion(actual, predicted, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    scenarios = sorted({label.split("_")[0] for label in LABELS})
    matrix = confusion_matrix([x.split("_")[0] for x in actual], [x.split("_")[0] for x in predicted], labels=scenarios)
    fig, ax = plt.subplots(figsize=(12,10))
    ConfusionMatrixDisplay(matrix, display_labels=scenarios).plot(ax=ax, cmap="Blues", xticks_rotation=90, values_format="d", colorbar=False)
    ax.set_title("Official test: scenario confusion (selected intent model)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
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
