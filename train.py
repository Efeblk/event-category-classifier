"""Train and evaluate event listing category classifiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from classifier import (
    ARTIFACT_TASK,
    DATASET_DOMAIN,
    LABELS,
    make_models,
    normalize_text,
)


REQUIRED_COLUMNS = ("id", "text", "label", "group_id", "title", "source", "source_url")
MIN_ROWS = 1_000
MIN_CLASS_ROWS_PER_SPLIT = 5
COMPARISON_ROWS_PER_CLASS = 12
DATASET_PROVENANCE = "gametime_public"
LANGUAGE_SUPPORT = ("en",)
BENCHMARK_NOTE = (
    "Metrics use held-out performer groups from public Gametime listings. Labels are "
    "the ticket site's categories and were not reviewed by hand."
)


def read_csv(path: str | Path, min_rows: int = MIN_ROWS) -> list[dict[str, str]]:
    """Read and validate the prepared CSV without changing its records."""
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1.")
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ValueError(f"Data file does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(missing)}")
        rows = [{name: (row.get(name) or "").strip() for name in REQUIRED_COLUMNS} for row in reader]

    if len(rows) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows. Found {len(rows)}.")

    ids = [row["id"] for row in rows]
    if any(not value for value in ids):
        raise ValueError("Every row needs a non-empty id.")
    duplicate_ids = [value for value, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate row ids found: {len(duplicate_ids)}")
    if any(not row["group_id"] for row in rows):
        raise ValueError("Every row needs a non-empty group_id.")
    if any(not normalize_text(row["text"]) for row in rows):
        raise ValueError("Every row needs non-empty text after normalization.")

    actual_labels = set(row["label"] for row in rows)
    if actual_labels != set(LABELS):
        raise ValueError(f"Labels must be exactly: {', '.join(LABELS)}")

    labels_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_group[row["group_id"]].add(row["label"])
    mixed_groups = sum(1 for labels in labels_by_group.values() if len(labels) > 1)
    if mixed_groups:
        raise ValueError(f"Found {mixed_groups} groups with more than one label.")

    normalized_counts = Counter(normalize_text(row["text"]) for row in rows)
    duplicate_texts = sum(1 for count in normalized_counts.values() if count > 1)
    if duplicate_texts:
        raise ValueError(
            f"Input is not deduplicated. Found {duplicate_texts} repeated normalized texts."
        )
    return rows



def _validate_split(rows: list[dict[str, str]], parts: dict[str, list[int]]) -> None:
    names = tuple(parts)
    group_sets = {
        name: {rows[index]["group_id"] for index in indices}
        for name, indices in parts.items()
    }
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise ValueError(f"Groups overlap between {left} and {right}.")

    all_indices = [index for indices in parts.values() for index in indices]
    if len(all_indices) != len(rows) or len(set(all_indices)) != len(rows):
        raise ValueError("Split rows are missing or duplicated.")

    for name, indices in parts.items():
        counts = Counter(rows[index]["label"] for index in indices)
        if set(counts) != set(LABELS):
            raise ValueError(f"The {name} split does not contain all classes.")
        if min(counts.values()) < MIN_CLASS_ROWS_PER_SPLIT:
            raise ValueError(
                f"The {name} split needs at least {MIN_CLASS_ROWS_PER_SPLIT} rows per class."
            )


def split_data(rows: list[dict[str, str]], seed: int = 42) -> dict[str, list[int]]:
    """Make deterministic, group-disjoint splits near 60/20/20."""
    from sklearn.model_selection import StratifiedGroupKFold

    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    groups = [row["group_id"] for row in rows]

    test_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_validation_indices, test_indices = next(
        test_splitter.split(texts, labels, groups)
    )
    remaining = list(map(int, train_validation_indices))
    validation_splitter = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=seed
    )
    relative_train, relative_validation = next(
        validation_splitter.split(
            [texts[index] for index in remaining],
            [labels[index] for index in remaining],
            [groups[index] for index in remaining],
        )
    )
    parts = {
        "train": [remaining[int(index)] for index in relative_train],
        "validation": [remaining[int(index)] for index in relative_validation],
        "test": list(map(int, test_indices)),
    }
    _validate_split(rows, parts)
    return parts


def cross_validate(
    rows: list[dict[str, str]], indices: list[int], seed: int = 42, folds: int = 5
) -> dict[str, dict[str, Any]]:
    """Report grouped k-fold macro F1 for fresh models. It does not select a model."""
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

    texts = [rows[index]["text"] for index in indices]
    labels = [rows[index]["label"] for index in indices]
    groups = [rows[index]["group_id"] for index in indices]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = {}
    for name, model in make_models(seed).items():
        scores = cross_val_score(
            model, texts, labels, groups=groups, cv=splitter, scoring="f1_macro"
        )
        results[name] = {
            "macro_f1_mean": float(scores.mean()),
            "macro_f1_std": float(scores.std()),
            "fold_macro_f1": [float(score) for score in scores],
        }
    return results


def _scores(actual: list[str], predicted: Iterable[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    prediction = list(predicted)
    report = classification_report(
        actual,
        prediction,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in LABELS
    }
    return {
        "accuracy": float(accuracy_score(actual, prediction)),
        "macro_f1": float(f1_score(actual, prediction, labels=list(LABELS), average="macro")),
        "per_class": per_class,
    }


def _predict_timed(model: Any, texts: list[str]) -> tuple[list[str], float]:
    started = time.perf_counter()
    predicted = [str(value) for value in model.predict(texts)]
    return predicted, time.perf_counter() - started


def _dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("scikit-learn", "numpy", "scipy", "joblib", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def comparison_records(
    rows: list[dict[str, str]], test_indices: list[int], seed: int = 42
) -> list[dict[str, str]]:
    """Pick a fixed, class-balanced sample of test titles for the three-method comparison."""
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index in sorted(test_indices, key=lambda index: rows[index]["id"]):
        by_label[rows[index]["label"]].append(rows[index])
    generator = random.Random(seed)
    records = []
    for label in LABELS:
        count = min(COMPARISON_ROWS_PER_CLASS, len(by_label[label]))
        for row in generator.sample(by_label[label], count):
            records.append({"id": row["id"], "text": row["text"], "label": label})
    return records


def _write_reports(
    report_dir: Path,
    rows: list[dict[str, str]],
    parts: dict[str, list[int]],
    evaluations: dict[str, dict[str, Any]],
    selected_name: str,
    selected_test_predictions: list[str],
    metadata: dict[str, Any],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    with (report_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    split_payload: dict[str, Any] = {"seed": metadata["seed"]}
    for name, indices in parts.items():
        split_payload[name] = {
            "row_ids": [rows[index]["id"] for index in indices],
            "group_ids": sorted({rows[index]["group_id"] for index in indices}),
        }
    with (report_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    comparison = {
        "name": "Gametime test-split comparison set",
        "provenance": "Sampled from the held-out test split. Gametime categories are the labels.",
        "license": "CC BY-NC 4.0",
        "records": comparison_records(rows, parts["test"], metadata["seed"]),
    }
    with (report_dir / "comparison_set.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    test_rows = [rows[index] for index in parts["test"]]
    with (report_dir / "errors.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("id", "group_id", "text", "actual", "predicted")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, prediction in zip(test_rows, selected_test_predictions):
            if row["label"] != prediction:
                writer.writerow({
                    "id": row["id"], "group_id": row["group_id"], "text": row["text"],
                    "actual": row["label"], "predicted": prediction,
                })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    actual = [row["label"] for row in test_rows]
    matrix = confusion_matrix(actual, selected_test_predictions, labels=list(LABELS))
    display = ConfusionMatrixDisplay(matrix, display_labels=list(LABELS))
    display.plot(cmap="Blues", values_format="d")
    display.ax_.set_title(f"Test confusion matrix: {selected_name}")
    display.figure_.tight_layout()
    display.figure_.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)



def train(
    data_path: str | Path = "data/events.csv",
    model_path: str | Path = "artifacts/model.joblib",
    report_dir: str | Path = "reports",
    seed: int = 42,
    allow_small_prototype: bool = False,
) -> dict[str, Any]:
    """Train candidates, select on validation, and evaluate once on test."""
    import joblib

    source_path = Path(data_path)
    rows = read_csv(source_path, min_rows=1 if allow_small_prototype else MIN_ROWS)
    parts = split_data(rows, seed)
    text_by_part = {
        name: [rows[index]["text"] for index in indices] for name, indices in parts.items()
    }
    label_by_part = {
        name: [rows[index]["label"] for index in indices] for name, indices in parts.items()
    }

    models = make_models(seed)
    evaluations: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(text_by_part["train"], label_by_part["train"])
        training_seconds = time.perf_counter() - started
        prediction, prediction_seconds = _predict_timed(model, text_by_part["validation"])
        evaluations[name] = {
            "training_seconds": training_seconds,
            "validation": {
                **_scores(label_by_part["validation"], prediction),
                "prediction_seconds": prediction_seconds,
                "batch_rows": len(prediction),
            },
        }

    # Cross-validation uses only train and validation rows. Test rows stay unseen.
    cross_validation = cross_validate(rows, parts["train"] + parts["validation"], seed)

    selected_name = max(
        models,
        key=lambda name: (evaluations[name]["validation"]["macro_f1"], name),
    )

    test_predictions: dict[str, list[str]] = {}
    for name, model in models.items():
        prediction, prediction_seconds = _predict_timed(model, text_by_part["test"])
        test_predictions[name] = prediction
        evaluations[name]["test"] = {
            **_scores(label_by_part["test"], prediction),
            "prediction_seconds": prediction_seconds,
            "batch_rows": len(prediction),
        }

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = _dataset_sha256(source_path)
    course_dataset_ready = len(rows) >= MIN_ROWS
    artifact = {
        "model": models[selected_name],
        "model_name": selected_name,
        "labels": sorted(LABELS),
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "task": ARTIFACT_TASK,
        "dataset_domain": DATASET_DOMAIN,
        "provenance": DATASET_PROVENANCE,
        "language_support": list(LANGUAGE_SUPPORT),
        "course_dataset_ready": course_dataset_ready,
    }
    joblib.dump(artifact, destination)

    split_counts = {
        name: {
            "rows": len(indices),
            "groups": len({rows[index]["group_id"] for index in indices}),
            "classes": dict(Counter(rows[index]["label"] for index in indices)),
        }
        for name, indices in parts.items()
    }
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "task": ARTIFACT_TASK,
        "dataset_domain": DATASET_DOMAIN,
        "provenance": DATASET_PROVENANCE,
        "language_support": list(LANGUAGE_SUPPORT),
        "course_dataset_ready": course_dataset_ready,
        "allow_small_prototype": allow_small_prototype,
        "evaluation_scope": "raw_model",
        "benchmark_note": BENCHMARK_NOTE,
        "dataset": {
            "path": str(source_path),
            "sha256": dataset_sha256,
            "rows": len(rows),
        },
        "environment": _package_versions(),
        "split": split_counts,
        "selection": {
            "metric": "validation_macro_f1",
            "selected_model": selected_name,
            "fit_split": "train",
            "final_refit": False,
            "test_used_for_selection": False,
        },
        "cross_validation": {
            "folds": 5,
            "rows": "train_and_validation",
            "used_for_selection": False,
            "models": cross_validation,
        },
        "saved_model": str(destination),
        "models": evaluations,
    }
    _write_reports(
        Path(report_dir), rows, parts, evaluations, selected_name,
        test_predictions[selected_name], metadata,
    )
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/events.csv")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-small-prototype",
        action="store_true",
        help="Allow fewer than 1000 rows for prototype training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        metadata = train(
            args.data,
            args.model,
            args.reports_dir,
            args.seed,
            allow_small_prototype=args.allow_small_prototype,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    selected = metadata["selection"]["selected_model"]
    score = metadata["models"][selected]["test"]["macro_f1"]
    print(f"selected_model={selected}")
    print(f"test_macro_f1={score:.4f}")
    print(f"model={metadata['saved_model']}")


if __name__ == "__main__":
    main(sys.argv[1:])
