"""Rebuild data and models in a temporary directory and verify exact evidence.

Run after prepare_data.py and train.py. Uses the already verified archive cache;
does not download data, contact Jev, or alter source data and trained artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from classifier import ARTIFACT_TASK, _validated_model, model_input
from prepare_data import SHA256, digest, prepare
from train import read_data, split_data, train, write_json


CLEANED_FILES = ("techniques.jsonl", "audit.json", "splits.json", "development.jsonl", "development_audit.json")
REPORT_FILES = ("metrics.json", "dev_examples.json", "comparison_set.json", "predictions.json")
ARTIFACT_FILES = ("model.joblib", "lr_model.joblib", "method_models.joblib")


def _compare_file(original: Path, fresh: Path) -> dict[str, Any]:
    original_bytes, fresh_bytes = original.read_bytes(), fresh.read_bytes()
    return {
        "original_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "fresh_sha256": hashlib.sha256(fresh_bytes).hexdigest(),
        "bytes_equal": original_bytes == fresh_bytes,
    }


def _compare_artifacts(original: Path, fresh: Path, original_inputs: list[dict],
                       fresh_inputs: list[dict]) -> dict[str, Any]:
    import joblib

    # These are artifacts just created locally by train.py, not arbitrary uploads.
    original_artifact, fresh_artifact = joblib.load(original), joblib.load(fresh)
    if original.name == "method_models.joblib":
        originals, fresh_models = original_artifact, fresh_artifact
        if not isinstance(originals, dict) or not isinstance(fresh_models, dict):
            raise ValueError("Method comparison artifact must be a dictionary.")
    else:
        originals, fresh_models = {original.name: original_artifact}, {fresh.name: fresh_artifact}
    if set(originals) != set(fresh_models):
        raise ValueError("Original and fresh artifacts contain different methods.")
    checks = {}
    for name in sorted(originals):
        original_model = _validated_model(originals[name])
        fresh_model = _validated_model(fresh_models[name])
        first = list(map(str, original_model.predict(original_inputs)))
        second = list(map(str, fresh_model.predict(fresh_inputs)))
        original_metadata = {key: value for key, value in originals[name].items() if key != "model"}
        fresh_metadata = {key: value for key, value in fresh_models[name].items() if key != "model"}
        checks[name] = {
            "model_name": originals[name]["model_name"],
            "metadata_equal": original_metadata == fresh_metadata,
            "dev_predictions_equal": first == second,
            "dev_prediction_count": len(first),
            "original_predictions_sha256": hashlib.sha256(json.dumps(first).encode("utf-8")).hexdigest(),
            "fresh_predictions_sha256": hashlib.sha256(json.dumps(second).encode("utf-8")).hexdigest(),
        }
    return checks


def reproduce(data_dir: str | Path = "data", artifacts_dir: str | Path = "artifacts",
              report_dir: str | Path = "reports", seed: int = 42) -> dict[str, Any]:
    """Run a fresh pipeline and record exact checks; failures leave a report."""
    source_data, source_artifacts, source_reports = Path(data_dir), Path(artifacts_dir), Path(report_dir)
    result: dict[str, Any] = {
        "schema_version": 1, "task": ARTIFACT_TASK, "seed": seed,
        "fresh_temporary_pipeline": True, "paid_calls": 0, "evaluation_scope": "dev_only",
        "checks": {}, "passed": False,
        "comparison": "Exact preserved and development cleaned/report bytes, every candidate's dev predictions, and saved model metadata/dev predictions.",
    }
    try:
        archive = source_data / "source" / "datasets-v2.tgz"
        if not archive.is_file():
            raise ValueError("Verified archive cache is missing; run prepare_data.py first.")
        archive_sha256 = digest(archive)
        if archive_sha256 != SHA256:
            raise ValueError("Cached archive SHA256 mismatch; reproduction refused.")
        required = [source_data / "cleaned" / name for name in CLEANED_FILES]
        required += [source_reports / name for name in REPORT_FILES]
        required += [source_artifacts / name for name in ARTIFACT_FILES]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError("Original pipeline outputs are missing: " + ", ".join(missing))
        original_rows = read_data(source_data / "cleaned" / "development.jsonl")
        original_indices = split_data(original_rows)["dev"]
        original_inputs = [model_input(original_rows[i]["text"], original_rows[i]["context"]) for i in original_indices]
        original_metrics = json.loads((source_reports / "metrics.json").read_text(encoding="utf-8"))
        if original_metrics.get("seed") != seed:
            raise ValueError("Reproduction seed differs from the original training run.")
        result["source_archive_sha256"] = archive_sha256
        with tempfile.TemporaryDirectory(prefix="spot-manipulation-reproduction-") as temporary:
            workspace = Path(temporary)
            fresh_data, fresh_artifacts, fresh_reports = workspace / "data", workspace / "artifacts", workspace / "reports"
            cache = fresh_data / "source" / "datasets-v2.tgz"
            cache.parent.mkdir(parents=True)
            shutil.copyfile(archive, cache)
            prepare(fresh_data, seed)
            train(fresh_data / "cleaned" / "development.jsonl", fresh_artifacts / "model.joblib", fresh_reports, seed)
            result["checks"]["cleaned_files"] = {
                name: _compare_file(source_data / "cleaned" / name, fresh_data / "cleaned" / name)
                for name in CLEANED_FILES
            }
            result["checks"]["reports"] = {
                name: _compare_file(source_reports / name, fresh_reports / name) for name in REPORT_FILES
            }
            first_predictions = json.loads((source_reports / "predictions.json").read_text(encoding="utf-8"))
            second_predictions = json.loads((fresh_reports / "predictions.json").read_text(encoding="utf-8"))
            if first_predictions.get("partition") != "dev" or second_predictions.get("partition") != "dev":
                raise ValueError("Prediction evidence must identify the development partition.")
            first_candidates, second_candidates = first_predictions.get("candidates"), second_predictions.get("candidates")
            if not isinstance(first_candidates, dict) or not isinstance(second_candidates, dict):
                raise ValueError("Prediction evidence must contain candidate dictionaries.")
            if set(first_candidates) != set(original_metrics["models"]) or set(first_candidates) != set(second_candidates):
                raise ValueError("Prediction evidence does not cover every original and fresh model candidate.")
            result["checks"]["candidate_predictions"] = {
                name: {"equal": first_candidates[name] == second_candidates[name],
                       "dev_prediction_count": len(first_candidates[name])}
                for name in sorted(first_candidates)
            }
            fresh_rows = read_data(fresh_data / "cleaned" / "development.jsonl")
            fresh_indices = split_data(fresh_rows)["dev"]
            fresh_inputs = [model_input(fresh_rows[i]["text"], fresh_rows[i]["context"]) for i in fresh_indices]
            result["checks"]["saved_artifacts"] = {
                name: _compare_artifacts(source_artifacts / name, fresh_artifacts / name, original_inputs, fresh_inputs)
                for name in ARTIFACT_FILES
            }
            byte_checks = [check["bytes_equal"] for group in ("cleaned_files", "reports")
                           for check in result["checks"][group].values()]
            candidate_checks = [check["equal"] for check in result["checks"]["candidate_predictions"].values()]
            artifact_checks = [check[key] for group in result["checks"]["saved_artifacts"].values()
                               for check in group.values() for key in ("metadata_equal", "dev_predictions_equal")]
            result["passed"] = all(byte_checks + candidate_checks + artifact_checks)
            if not result["passed"]:
                result["error"] = "Fresh pipeline outputs differ from original evidence; inspect failed checks."
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        result["error"] = str(error)
    write_json(source_reports / "reproduction_check.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = reproduce(args.data_dir, args.artifacts_dir, args.reports_dir, args.seed)
    print(json.dumps({"passed": result["passed"], "report": str(Path(args.reports_dir) / "reproduction_check.json"),
                      **({"error": result["error"]} if "error" in result else {})}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
