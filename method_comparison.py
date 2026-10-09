"""Compare independently selected local models on the same excerpt and context."""

from __future__ import annotations

import time
from typing import Any

from classifier import _validated_model, classify_request


METHODS = ("naive_bayes", "logistic_regression", "linear_svm")
MAX_TEXT_LENGTH = 5000
MAX_CONTEXT_LENGTH = 10000


def validate_input(text: object, context: object = "", include_jev: object = False) -> None:
    """Reject invalid inputs before running any model or paid provider."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_LENGTH:
        raise ValueError("Enter an excerpt with 1 to 5,000 characters.")
    if not isinstance(context, str) or len(context) > MAX_CONTEXT_LENGTH:
        raise ValueError("Context must be text with at most 10,000 characters.")
    if type(include_jev) is not bool:
        raise ValueError("include_jev must be true or false.")


def validate_artifacts(artifacts: object) -> dict[str, Any]:
    """Validate three fitted, compatible artifacts from the same training run."""
    if not isinstance(artifacts, dict) or set(artifacts) != set(METHODS):
        raise ValueError("Comparison needs Naive Bayes, Logistic Regression and Linear SVM artifacts.")
    families = {
        "naive_bayes": ("multinomial_nb_", "naive_bayes_"),
        "logistic_regression": ("logistic_regression_",),
        "linear_svm": ("linear_svm_",),
    }
    for method in METHODS:
        artifact = artifacts[method]
        _validated_model(artifact)
        if not artifact["model_name"].startswith(families[method]):
            raise ValueError(f"The {method} entry contains the wrong model family.")
    for field, description in (("dataset_sha256", "training data"), ("code_sha256", "training code"), ("seed", "training seed")):
        if len({artifact[field] for artifact in artifacts.values()}) != 1:
            raise ValueError(f"Comparison artifacts must use the same {description}.")
    for field in ("context_margin", "evaluation_scope"):
        if len({artifact["provenance"].get(field) for artifact in artifacts.values()}) != 1:
            raise ValueError(f"Comparison artifacts must use the same provenance {field}.")
    return artifacts


def compare_request(
    artifacts: object,
    text: object,
    jev: Any = None,
    include_jev: bool = False,
    context: object = "",
) -> dict[str, Any]:
    """Run every method independently on exactly the original input strings."""
    validate_input(text, context, include_jev)
    models = validate_artifacts(artifacts)
    results = []
    for method in METHODS:
        started = time.perf_counter()
        prediction = classify_request(models[method], text, context=context)
        results.append({
            **prediction,
            "method": method,
            "model": models[method]["model_name"],
            "status": "ok",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        })
    provider = {
        "status": "not_requested",
        "message": "Jev was not requested. No API call was made.",
    }
    if include_jev:
        provider = jev.classify(text, context=context) if jev is not None else {
            "status": "not_configured",
            "message": "Jev is not configured. No API call was made.",
        }
    results.append({**provider, "method": "jev"})
    return {
        "results": results,
        "models": {method: models[method]["model_name"] for method in METHODS},
        "scope": "Technique classification of a selected English excerpt with optional context. "
                 "This does not assess truth or detect propaganda throughout a whole article.",
    }
