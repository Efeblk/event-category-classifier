"""Classify the same input with rules, Logistic Regression, and optional Jev."""

import time

from classifier import _validated_model, classify_request
from parser import parse_request


def compare_request(artifact, text, jev=None, include_jev=False):
    if not isinstance(text, str) or not text.strip() or len(text) > 5000:
        raise ValueError("Enter a request with 1 to 5,000 characters.")
    if type(include_jev) is not bool:
        raise ValueError("include_jev must be true or false.")
    _validated_model(artifact)
    if not artifact["model_name"].startswith("logistic_regression_"):
        raise ValueError("The three-method comparison requires a Logistic Regression artifact.")
    results = []
    for key, predict in (("parser", lambda: {"label": parse_request(text)}),
                         ("logistic_regression", lambda: classify_request(artifact, text))):
        started = time.perf_counter()
        prediction = predict()
        results.append({"method": key, "status": "ok", "label": prediction["label"],
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)})
    result = {"status": "not_requested", "message": "Jev was not requested. No API call was made."}
    if include_jev:
        result = jev.classify(text) if jev is not None else {
            "status": "not_configured", "message": "Jev is not configured. No API call was made."}
    results.append({"method": "jev", **result})
    return {"results": results, "model": artifact["model_name"],
            "scope": "Same original input. Parser uses rules. Logistic Regression has no language guards; zero-vocabulary input returns unclear."}
