"""Run one prediction with a trusted local model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from request_policy import classify_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Turkish or English user request")
    parser.add_argument(
        "--model",
        default="artifacts/model.joblib",
        help="Trusted local joblib model path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"Model file does not exist: {model_path}")

    import joblib

    artifact = joblib.load(model_path)
    try:
        result = classify_input(artifact, args.text)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
