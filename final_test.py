"""One-time evaluation of the development-only upgrade on the old custom test.

The old test errors informed the upgrade's design, so the score is optimistically
biased and is not a fresh independent final evaluation. Trained artifacts are
loaded from trusted local joblib files and only predict; nothing is refit. The
dev reports/metrics.json keeps every test score null.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from classifier import ARTIFACT_TASK, DATASET_DOMAIN, _validated_model, model_input, normalize_text
from json_validation import strict_json
from method_comparison import METHODS, validate_artifacts
from prepare_data import (
    DEVELOPMENT_CONTEXT_MARGIN, PARTITIONS, SOURCE_PARTITION, digest, normalize_whitespace, read_annotation_file,
)
from train import plot_confusion, scores, top_confusions, write_json


REPORT_SCHEMA_VERSION = 1
EVALUATION_SCOPE = "old_test_optimistic_one_time"
CONTEXT_MARGIN = 1000
CAVEAT = (
    "Optimistically biased, one-time evaluation of the old custom test partition. The old test "
    "errors informed the upgrade's design (features, context and model choice), so these scores "
    "are not a fresh independent final evaluation. Authorized by the user on 2026-10-10. The dev "
    "reports/metrics.json still keeps every test score null."
)
HISTORICAL_BASELINE = {
    "source": "evidence/baseline/metrics.json",
    "selected_model": "logistic_regression_word_context",
    "test_accuracy": 0.47315175097276263,
    "test_macro_f1": 0.35416699695753623,
    "note": "Original pipeline (350-character context, nine candidates) on the same old custom test: 47.32% accuracy, 0.3542 macro F1.",
}
SCOPE_REQUIREMENTS = {"evaluation_scope": "dev_only", "context_margin": CONTEXT_MARGIN}


def build_test_rows(data_dir: str | Path = "data") -> list[dict]:
    """Reconstruct the old custom test rows with the upgrade's 1,000-character context.

    Used only here for the one-time, optimistically biased old-test
    evaluation. Nothing is written under data/cleaned/; the returned rows are not
    training data and must never be fitted or used for selection.
    """
    directory = Path(data_dir)
    manifest_path = directory / "cleaned" / "splits.json"
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError("Test evaluation needs the original valid split manifest.") from error
    if not isinstance(manifest, dict) or manifest.get("source_partition") != SOURCE_PARTITION:
        raise ValueError("Test evaluation requires the official-training source partition.")
    assignments = manifest.get("article_partitions")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("Test evaluation needs article partition assignments.")
    test_articles = []
    for article_id, partition in assignments.items():
        if not isinstance(article_id, str) or not article_id.isascii() or not article_id.isdigit():
            raise ValueError("Split article IDs must contain ASCII digits only.")
        if partition not in PARTITIONS:
            raise ValueError("Split manifest contains an unsupported partition.")
        if partition == "test":
            test_articles.append(article_id)
    expected_rows = (manifest.get("row_counts") or {}).get("test")
    expected_articles = (manifest.get("article_counts") or {}).get("test")
    if type(expected_rows) is not int or expected_rows < 1 or type(expected_articles) is not int:
        raise ValueError("Split manifest needs positive integer test row and article counts.")
    if len(test_articles) != expected_articles:
        raise ValueError("Test article count differs from the original split manifest.")
    rows, seen = [], set()
    for article_id in sorted(test_articles):
        article_path = directory / "raw" / "articles" / f"article{article_id}.txt"
        annotation_path = directory / "raw" / "annotations" / f"article{article_id}.task2-TC.labels"
        try:
            article = article_path.read_bytes().decode("utf-8")
        except (OSError, ValueError) as error:
            raise ValueError(f"Unable to read UTF-8 test source article {article_id}.") from error
        for annotation in read_annotation_file(annotation_path, article_id, article):
            start, end, label = annotation["start"], annotation["end"], annotation["label"]
            identity = (article_id, start, end, label)
            if identity in seen:
                continue
            seen.add(identity)
            text = article[start:end]
            context = normalize_whitespace(article[max(0, start - DEVELOPMENT_CONTEXT_MARGIN):
                                                   min(len(article), end + DEVELOPMENT_CONTEXT_MARGIN)])
            rows.append({"id": f"{article_id}:{start}:{end}:{label}",
                         "article_id": article_id, "text": text,
                         "clean_text": normalize_whitespace(text), "context": context,
                         "label": label, "start": start, "end": end,
                         "partition": "test", "source_partition": SOURCE_PARTITION})
    rows.sort(key=lambda row: (row["article_id"], row["start"], row["end"], row["label"]))
    if len(rows) != expected_rows:
        raise ValueError(f"Test rows ({len(rows)}) differ from the original split manifest ({expected_rows}).")
    return rows


def _check_upgrade_artifact(artifact: Any, description: str) -> None:
    provenance = artifact["provenance"]
    for key, expected in SCOPE_REQUIREMENTS.items():
        if provenance.get(key) != expected:
            raise ValueError(f"{description} must be a development-only artifact with {key}={expected!r}.")


def load_artifacts(models_path: str | Path, model_path: str | Path) -> tuple[dict[str, Any], Any]:
    """Load trusted local artifacts and verify they are one development-only run."""
    import joblib

    method_artifacts = validate_artifacts(joblib.load(models_path))
    overall = joblib.load(model_path)
    _validated_model(overall)
    for method, artifact in method_artifacts.items():
        _check_upgrade_artifact(artifact, f"Method model {method}")
    _check_upgrade_artifact(overall, "Overall model")
    reference = method_artifacts[METHODS[0]]
    for field in ("dataset_sha256", "code_sha256", "seed"):
        if overall[field] != reference[field]:
            raise ValueError(f"Overall model {field} differs from the method models; rerun train.py.")
    return method_artifacts, overall


def check_metrics_file(path: str | Path, overall: Any, method_artifacts: dict[str, Any]) -> dict[str, Any] | None:
    """Confirm the dev report belongs to these artifacts when it is present."""
    metrics_path = Path(path)
    if not metrics_path.is_file():
        return None
    metrics = strict_json(metrics_path.read_bytes())
    if not isinstance(metrics, dict):
        raise ValueError("Development metrics must be a JSON object.")
    dataset = metrics.get("dataset") or {}
    selection = metrics.get("selection") or {}
    if dataset.get("sha256") != overall["dataset_sha256"] or metrics.get("code_sha256") != overall["code_sha256"]:
        raise ValueError("reports/metrics.json does not match the artifact dataset or code hash; rerun train.py.")
    expected_methods = {method: artifact["model_name"] for method, artifact in method_artifacts.items()}
    if selection.get("selected_model") != overall["model_name"] or selection.get("method_models") != expected_methods:
        raise ValueError("reports/metrics.json selects different models than the loaded artifacts; rerun train.py.")
    return {"selected_model": selection["selected_model"], "method_models": expected_methods}


def dev_overlap(test_rows: list[dict[str, Any]], development_path: str | Path) -> dict[str, int]:
    """Count test articles and normalized excerpt/context pairs seen in train/dev."""
    development = [strict_json(line) for line in Path(development_path).read_bytes().splitlines() if line.strip()]
    development_articles = {row["article_id"] for row in development}
    development_pairs = {(normalize_text(row["text"]), normalize_text(row["context"])) for row in development}
    test_articles = {row["article_id"] for row in test_rows}
    if test_articles & development_articles:
        raise ValueError("A test article also appears in train/dev; the split is invalid.")
    pairs = [(normalize_text(row["text"]), normalize_text(row["context"])) for row in test_rows]
    return {
        "test_articles_in_train_or_dev": len(test_articles & development_articles),
        "normalized_pairs_also_in_development": len({pair for pair in pairs if pair in development_pairs}),
        "test_rows_with_development_pair": sum(pair in development_pairs for pair in pairs),
    }


def majority_label(development_path: str | Path) -> str:
    """Most frequent train label; ties break alphabetically for determinism."""
    counts = Counter(row["label"] for row in
                     (strict_json(line) for line in Path(development_path).read_bytes().splitlines() if line.strip())
                     if row["partition"] == "train")
    if not counts:
        raise ValueError("Majority baseline needs train rows in development.jsonl.")
    return min(counts, key=lambda label: (-counts[label], label))


def run_evaluation(
    models_path: str | Path = "artifacts/method_models.joblib",
    model_path: str | Path = "artifacts/model.joblib",
    data_dir: str | Path = "data",
    report_path: str | Path = "reports/test_evaluation.json",
    metrics_path: str | Path = "reports/metrics.json",
    force: bool = False,
) -> dict[str, Any]:
    """Score the saved development-only models once on the old test partition."""
    report = Path(report_path)
    figure = report.with_name("test_confusion_matrix.png")
    if (report.exists() or figure.exists()) and not force:
        raise ValueError(f"{report} or {figure} already exists. The old-test evaluation is one-time; use --force only to deliberately replace it.")
    method_artifacts, overall = load_artifacts(models_path, model_path)
    metrics_check = check_metrics_file(metrics_path, overall, method_artifacts)
    data = Path(data_dir)
    development_path = data / "cleaned" / "development.jsonl"
    test_rows = build_test_rows(data)
    overlap = dev_overlap(test_rows, development_path)
    actual = [row["label"] for row in test_rows]
    inputs = [model_input(row["text"], row["context"]) for row in test_rows]
    candidates = {"selected_overall": ("overall development winner", overall)}
    candidates.update({method: (f"dev-selected {method} model", artifact)
                       for method, artifact in method_artifacts.items()})
    models, selected_predictions = {}, None
    for key, (role, artifact) in candidates.items():
        predicted = list(map(str, artifact["model"].predict(inputs)))
        if key == "selected_overall":
            selected_predictions = predicted
        models[key] = {"role": role, "model_name": artifact["model_name"], **scores(actual, predicted)}
    baseline_label = majority_label(development_path)
    majority = {"label": baseline_label, **scores(actual, [baseline_label] * len(actual))}
    figure.parent.mkdir(parents=True, exist_ok=True)
    plot_confusion(actual, selected_predictions, figure, partition="test")
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "task": ARTIFACT_TASK,
        "dataset_domain": DATASET_DOMAIN,
        "evaluation_scope": EVALUATION_SCOPE,
        "partition": "test",
        "rows": len(test_rows),
        "articles": len({row["article_id"] for row in test_rows}),
        "context_margin": CONTEXT_MARGIN,
        "seed": overall["seed"],
        "dataset_sha256": overall["dataset_sha256"],
        "code_sha256": overall["code_sha256"],
        "artifacts": {
            "overall": {"path": Path(model_path).as_posix(), "sha256": digest(model_path),
                        "model_name": overall["model_name"]},
            "method_models": {"path": Path(models_path).as_posix(), "sha256": digest(models_path),
                              "models": {method: artifact["model_name"] for method, artifact in method_artifacts.items()}},
        },
        "development_metrics_check": metrics_check,
        "models": models,
        "majority_baseline": majority,
        "top_confusions_selected": top_confusions(actual, selected_predictions),
        "dev_test_overlap": overlap,
        "historical_baseline_test": HISTORICAL_BASELINE,
        "figure": figure.as_posix(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveat": CAVEAT,
    }
    write_json(report, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="artifacts/method_models.joblib")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--report", default="reports/test_evaluation.json")
    parser.add_argument("--force", action="store_true", help="Replace an existing one-time report.")
    arguments = parser.parse_args()
    try:
        result = run_evaluation(arguments.models, arguments.model, arguments.data_dir,
                                arguments.report, force=arguments.force)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for key, model in result["models"].items():
        print(f"{key} ({model['model_name']}): accuracy {model['accuracy']:.4f}, macro F1 {model['macro_f1']:.4f}")
    print(f"majority baseline: accuracy {result['majority_baseline']['accuracy']:.4f}, "
          f"macro F1 {result['majority_baseline']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
