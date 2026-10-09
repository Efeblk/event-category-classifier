"""Train persuasion classifiers with development-only or historical baseline scoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from classifier import (
    ARTIFACT_TASK, ARTIFACT_VERSION, DATASET_DOMAIN, LABELS,
    make_models, model_input, normalize_text,
)


MIN_ROWS = 1000
PARTITIONS = ("train", "dev")
BASELINE_PARTITIONS = PARTITIONS + ("test",)
METHOD_PREFIXES = {
    "naive_bayes": "multinomial_nb_",
    "logistic_regression": "logistic_regression_",
    "linear_svm": "linear_svm_",
}
BENCHMARK = {
    "source": "https://aclanthology.org/2020.semeval-1.186/",
    "paper": "Da San Martino et al. (2020), SemEval-2020 Task 11",
    "reference": "Corrected technique-classification ranking: Appendix B, Table 9, page 1405.",
    "quoted_official_test_micro_f1": {"ApplicaAI": 0.6374, "length_only_logistic_regression_baseline": 0.2520},
    "directly_comparable": False,
    "reason": "Our held-out articles come from official training data; the official test gold is not in this archive. Paper systems use a different test set and may use pretrained neural models.",
    "official_metric": "micro F1; our additional primary selection metric is fixed-14-label macro F1",
}


def read_data(path: str | Path, min_rows: int = MIN_ROWS, profile: str = "development") -> list[dict[str, Any]]:
    """Validate prepared annotations without changing excerpts or gold labels."""
    if profile not in ("development", "baseline"):
        raise ValueError("Training profile must be development or baseline.")
    partitions = BASELINE_PARTITIONS if profile == "baseline" else PARTITIONS
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1.")
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Data file does not exist: {path}")
    rows = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on data line {number}.") from error
    if len(rows) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows. Found {len(rows)}.")
    required = {"id", "article_id", "text", "context", "label", "start", "end", "partition", "source_partition"}
    ids, annotations = set(), set()
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Missing prepared data fields on row {number}.")
        if row["partition"] not in partitions:
            raise ValueError("Unsupported partition for this training profile; development excludes test rows.")
        if not isinstance(row["id"], str) or not row["id"].strip() or row["id"] in ids:
            raise ValueError("Empty or duplicate row id.")
        ids.add(row["id"])
        if not isinstance(row["article_id"], str) or not row["article_id"].strip():
            raise ValueError("Every row needs a non-empty article id.")
        for field in ("text", "context"):
            if not isinstance(row[field], str) or not normalize_text(row[field]):
                raise ValueError(f"Every row needs non-empty {field}.")
        if row["label"] not in LABELS:
            raise ValueError("Unsupported persuasion technique label.")
        if row["source_partition"] != "official_train":
            raise ValueError("Unsupported partition or source partition.")
        start, end = row["start"], row["end"]
        if any(not isinstance(offset, int) or isinstance(offset, bool) for offset in (start, end)):
            raise ValueError("Excerpt offsets must be integers.")
        if start < 0 or end <= start or len(row["text"]) != end - start:
            raise ValueError("Excerpt text does not match its character-offset length.")
        annotation = (row["article_id"], start, end, row["label"])
        if annotation in annotations:
            raise ValueError("Exact duplicate annotations must be removed by preprocessing.")
        annotations.add(annotation)
    return rows


def split_data(rows: list[dict[str, Any]], seed: int = 42, profile: str = "development") -> dict[str, list[int]]:
    """Validate already prepared groups; never resplit or tune on labels here."""
    del seed  # Preparation owns the deterministic split assignment.
    if profile not in ("development", "baseline"):
        raise ValueError("Training profile must be development or baseline.")
    partitions = BASELINE_PARTITIONS if profile == "baseline" else PARTITIONS
    parts = {part: [i for i, row in enumerate(rows) if row["partition"] == part] for part in partitions}
    if sum(map(len, parts.values())) != len(rows):
        raise ValueError("Unsupported partition.")
    for part, indices in parts.items():
        if not indices:
            raise ValueError(f"The {part} split is empty.")
    if {rows[i]["label"] for i in parts["train"]} != set(LABELS):
        raise ValueError("The train split must contain all 14 technique labels.")
    article_parts, input_parts = defaultdict(set), defaultdict(set)
    for row in rows:
        article_parts[row["article_id"]].add(row["partition"])
        pair = (normalize_text(row["text"]), normalize_text(row["context"]))
        input_parts[pair].add(row["partition"])
    if any(len(found) > 1 for found in article_parts.values()):
        raise ValueError("An article crosses train/dev/test partitions.")
    if any(len(found) > 1 for found in input_parts.values()):
        raise ValueError("An identical normalized excerpt/context crosses partitions.")
    return parts


def scores(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    """Score annotation rows using a fixed 14-label denominator."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    labels = list(LABELS)
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Scoring needs equally sized, non-empty gold and predicted labels.")
    if any(label not in LABELS for label in actual + predicted):
        raise ValueError("Scoring received an unsupported label.")
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "micro_f1": float(f1_score(actual, predicted, labels=labels, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "confusion_labels": labels,
    }


def select_model(evaluations: dict[str, Any], names: list[str] | None = None) -> str:
    """Use dev macro F1 only, with a deterministic alphabetical tie break."""
    candidates = list(evaluations) if names is None else names
    if not candidates:
        raise ValueError("No model candidates were provided for selection.")
    return min(candidates, key=lambda name: (-evaluations[name]["dev"]["macro_f1"], name))


def _record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("id", "article_id", "text", "context", "label")}


def comparison_records(rows: list[dict[str, Any]], indices: list[int], seed: int = 42,
                       partition: str = "dev") -> list[dict[str, Any]]:
    """Choose a uniform sample from the profile's declared evaluation partition."""
    if partition not in ("dev", "test") or any(rows[i]["partition"] != partition for i in indices):
        raise ValueError(f"Comparison records must come from {partition} only.")
    indices = sorted(indices, key=lambda i: rows[i]["id"])
    sample = random.Random(seed).sample(indices, min(40, len(indices)))
    return [_record(rows[i]) for i in sample]


def dev_records(rows: list[dict[str, Any]], dev_indices: list[int], seed: int = 42) -> list[dict[str, Any]]:
    """Choose one available dev example per label, without looking at predictions."""
    if any(rows[i]["partition"] != "dev" for i in dev_indices):
        raise ValueError("Demo examples must come from dev only.")
    by_label = defaultdict(list)
    for i in sorted(dev_indices, key=lambda i: rows[i]["id"]):
        by_label[rows[i]["label"]].append(i)
    generator = random.Random(seed)
    return [_record(rows[generator.choice(by_label[label])]) for label in LABELS if by_label[label]]


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def duplicate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit repeated lexical inputs and genuine multiple technique annotations."""
    text_parts, text_labels, pair_parts, span_labels = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rows:
        text = normalize_text(row["text"])
        text_parts[text].add(row["partition"])
        text_labels[text].add(row["label"])
        pair_parts[(text, normalize_text(row["context"]))].add(row["partition"])
        span_labels[(row["article_id"], row["start"], row["end"])].add(row["label"])
    return {
        "cross_partition_excerpt_texts": sum(len(found) > 1 for found in text_parts.values()),
        "cross_partition_excerpt_context_pairs": sum(len(found) > 1 for found in pair_parts.values()),
        "mixed_label_excerpt_texts": sum(len(found) > 1 for found in text_labels.values()),
        "multi_label_exact_spans": sum(len(found) > 1 for found in span_labels.values()),
        "action": "Retain distinct gold annotations; exact duplicates removed in preparation; identical excerpt/context grouped together.",
    }


def top_confusions(actual: list[str], predicted: list[str], count: int = 15) -> list[dict[str, Any]]:
    pairs = Counter((gold, guess) for gold, guess in zip(actual, predicted) if gold != guess)
    return [{"gold": gold, "predicted": guess, "count": n} for (gold, guess), n in pairs.most_common(count)]


def plot_confusion(actual: list[str], predicted: list[str], path: Path, partition: str = "dev") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    short_labels = [
        "Authority", "Fear/prejudice", "Bandwagon/Hitler", "Black-and-white",
        "Simple causality", "Doubt", "Exaggeration/minimisation", "Flag-waving",
        "Loaded language", "Name calling", "Repetition", "Slogans",
        "Thought-ending cliche", "Whataboutism/straw man/red herring",
    ]
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(confusion_matrix(actual, predicted, labels=list(LABELS)), display_labels=short_labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=90, values_format="d", colorbar=False,
    )
    ax.set_title(f"{partition.title()} articles: technique confusion (dev-selected model)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _code_hashes() -> tuple[str, dict[str, str]]:
    directory = Path(__file__).resolve().parent
    names = ("classifier.py", "train.py", "prepare_data.py")
    payload = b""
    hashes = {}
    for name in names:
        path = directory / name
        if path.is_file():
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            payload += name.encode("utf-8") + b"\0" + content + b"\0"
    return hashlib.sha256(payload).hexdigest(), hashes


def train(
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    report_dir: str | Path | None = None,
    seed: int = 42,
    allow_small_prototype: bool = False,
    profile: str = "development",
) -> dict[str, Any]:
    import joblib

    if profile not in ("development", "baseline"):
        raise ValueError("Training profile must be development or baseline.")
    historical = profile == "baseline"
    data_path = data_path or ("data/cleaned/techniques.jsonl" if historical else "data/cleaned/development.jsonl")
    model_path = model_path or ("artifacts/baseline/model.joblib" if historical else "artifacts/model.joblib")
    report_dir = report_dir or ("reports/baseline" if historical else "reports")
    evaluation_partition = "test" if historical else "dev"
    evaluation_scope = "historical_baseline" if historical else "dev_only"
    context_margin = 350 if historical else 1000
    rows = read_data(data_path, min_rows=1 if allow_small_prototype else MIN_ROWS, profile=profile)
    parts = split_data(rows, profile=profile)
    inputs = {part: [model_input(rows[i]["text"], rows[i]["context"]) for i in indices] for part, indices in parts.items()}
    labels = {part: [rows[i]["label"] for i in indices] for part, indices in parts.items()}
    models, evaluations, dev_predictions = make_models(seed, profile=profile), {}, {}
    for name, model in models.items():
        print(f"fit {name}", flush=True)
        model.fit(inputs["train"], labels["train"])
        dev_predictions[name] = list(map(str, model.predict(inputs["dev"])))
        evaluations[name] = {"dev": scores(labels["dev"], dev_predictions[name]), "test": None}

    # All choices are frozen before any candidate sees a test input.
    selected = select_model(evaluations)
    method_models = {
        method: select_model(evaluations, [name for name in models if name.startswith(prefix)])
        for method, prefix in METHOD_PREFIXES.items()
    }
    print(f"dev selection frozen: {selected}", flush=True)
    predictions = dev_predictions
    if historical:
        predictions = {}
        for name, model in models.items():
            predictions[name] = list(map(str, model.predict(inputs["test"])))
            evaluations[name]["test"] = scores(labels["test"], predictions[name])

    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    code_digest, script_hashes = _code_hashes()
    provenance = {
        "dataset": "SemEval-2020 Task 11 Propaganda Techniques Corpus",
        "source_partition": "official_train",
        "split_method": "seeded article/duplicate-input groups prepared before training; custom 65/15/20 target",
        "fit_partition": "train",
        "evaluation_scope": evaluation_scope,
        "context_margin": context_margin,
        "evaluation_unit": "one annotation row; distinct overlapping/multiple gold techniques retained",
    }

    def artifact(name: str) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_VERSION, "model": models[name], "model_name": name,
            "labels": LABELS, "dataset_sha256": digest, "code_sha256": code_digest,
            "seed": seed, "task": ARTIFACT_TASK, "dataset_domain": DATASET_DOMAIN,
            "provenance": provenance,
        }

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact(selected), destination)
    joblib.dump(artifact(method_models["logistic_regression"]), destination.with_name("lr_model.joblib"))
    joblib.dump({method: artifact(name) for method, name in method_models.items()}, destination.with_name("method_models.joblib"))
    metadata = {
        "schema_version": ARTIFACT_VERSION, "seed": seed, "task": ARTIFACT_TASK, "profile": profile,
        "dataset_domain": DATASET_DOMAIN,
        "dataset": {"sha256": digest, "rows": len(rows), "articles": len({row["article_id"] for row in rows}), "source_partition": "official_train"},
        "code_sha256": code_digest, "script_sha256": script_hashes,
        "partitions": {
            part: {
                "rows": len(indices), "articles": len({rows[i]["article_id"] for i in indices}),
                "label_counts": dict(Counter(labels[part])), "missing_labels": sorted(set(LABELS) - set(labels[part])),
            } for part, indices in parts.items()
        },
        "duplicate_audit": duplicate_audit(rows),
        "selection": {
            "metric": "dev_macro_f1", "selected_model": selected,
            "logistic_regression_model": method_models["logistic_regression"], "method_models": method_models,
            "fit_split": "train", "test_used_for_selection": False,
            "tie_break": "alphabetically first model name", "all_choices_frozen_before_test": True,
            "development_is_selection_data": True,
        },
        "evaluation": {
            "scope": evaluation_scope, "partition": evaluation_partition, "context_margin": context_margin,
            "unit": "annotation row", "labels": list(LABELS), "absent_classes_macro_f1": "zero",
            "micro_f1_equals_accuracy": True, "test_predictions_per_candidate": 1 if historical else 0,
            "independent_new_test_evaluation": False,
            "span_identification": False, "truth_verification": False,
        },
        "benchmark": BENCHMARK, "models": evaluations,
        "top_confused_technique_pairs": top_confusions(labels[evaluation_partition], predictions[selected]),
    }
    directory = Path(report_dir)
    write_json(directory / "metrics.json", metadata)
    write_json(directory / "predictions.json", {
        "partition": evaluation_partition,
        "candidates": {
            name: [{"id": rows[i]["id"], "gold": rows[i]["label"], "predicted": predicted}
                   for i, predicted in zip(parts[evaluation_partition], candidate_predictions)]
            for name, candidate_predictions in predictions.items()
        },
    })
    report_provenance = {
        "training_data_sha256": digest, "training_code_sha256": code_digest,
        "training_seed": seed, "model_evaluation_scope": evaluation_scope,
        "context_margin": context_margin,
    }
    write_json(directory / "dev_examples.json", {**report_provenance, "partition": "dev", "selection": "seeded one example per available label; no prediction filtering", "records": dev_records(rows, parts["dev"], seed)})
    write_json(directory / "comparison_set.json", {
        **report_provenance,
        "name": f"40 {evaluation_partition} persuasion annotations", "partition": evaluation_partition,
        "provenance": f"Uniform {evaluation_partition} sample without replacement, seed {seed}; same input/context for every method. Development is selection data, not an independent new test.",
        "records": comparison_records(rows, parts[evaluation_partition], seed, evaluation_partition),
    })
    with (directory / "errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "article_id", "text", "context", "gold", "predicted"])
        for i, predicted in zip(parts[evaluation_partition], predictions[selected]):
            row = rows[i]
            if row["label"] != predicted:
                writer.writerow([row["id"], row["article_id"], row["text"], row["context"], row["label"], predicted])
    plot_confusion(labels[evaluation_partition], predictions[selected], directory / "confusion_matrix.png", evaluation_partition)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--reports-dir")
    parser.add_argument("--profile", choices=("development", "baseline"), default="development")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-small-prototype", action="store_true")
    args = parser.parse_args()
    try:
        result = train(args.data, args.model, args.reports_dir, args.seed, args.allow_small_prototype, args.profile)
    except ValueError as error:
        parser.error(str(error))
    print(result["selection"])


if __name__ == "__main__":
    main()
