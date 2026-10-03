"""Run one prediction with a trusted local model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from classifier import normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Turkish event title and description")
    parser.add_argument(
        "--model",
        default="artifacts/model.joblib",
        help="Trusted local joblib model path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not normalize_text(args.text):
        raise SystemExit("Prediction text must contain non-whitespace characters.")
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"Model file does not exist: {model_path}")

    import joblib

    artifact = joblib.load(model_path)
    required_keys = {"model", "model_name", "labels", "dataset_sha256", "seed"}
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise SystemExit("Model artifact has an unsupported schema.")
    model = artifact["model"]
    if sorted(map(str, model.classes_)) != sorted(map(str, artifact["labels"])):
        raise SystemExit("Model classes do not match artifact labels.")
    label = str(model.predict([args.text])[0])
    result: dict[str, object] = {"label": label}

    # Only estimators with native predict_proba supply a probability score.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([args.text])[0]
        classes = list(model.classes_)
        result["score"] = round(float(probabilities[classes.index(label)]), 6)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
