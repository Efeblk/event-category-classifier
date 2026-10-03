"""Text normalization, model definitions, and request classification."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


LABELS = ("concert", "theatre", "stand_up", "unclear")
ARTIFACT_TASK = "user_request_classification"
DATASET_DOMAIN = "user_requests"


_WHITESPACE = re.compile(r"\s+")
_TURKISH_UPPER_I = str.maketrans({"I": "\u0131", "\u0130": "i"})
_TURKISH_LETTER = re.compile("[çğıöşüÇĞİÖŞÜ]")


def normalize_text(value: object) -> str:
    """Normalize Unicode, Turkish or English case, and whitespace."""
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    if _TURKISH_LETTER.search(text):
        text = text.translate(_TURKISH_UPPER_I).lower()
    else:
        text = text.replace("\u0130", "i").lower()
    return _WHITESPACE.sub(" ", text).strip()


def make_models(seed: int = 42) -> dict[str, Any]:
    """Create fresh candidate pipelines. Imports stay local for data preparation."""
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    def pipeline(
        classifier: Any,
        ngrams: tuple[int, int],
        analyzer: str = "word",
    ) -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=normalize_text,
                        lowercase=False,
                        analyzer=analyzer,
                        ngram_range=ngrams,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", classifier),
            ]
        )

    return {
        "dummy_most_frequent": pipeline(
            DummyClassifier(strategy="most_frequent", random_state=seed), (1, 1)
        ),
        "multinomial_nb": pipeline(MultinomialNB(alpha=1.0), (1, 2)),
        "logistic_regression_unigram": pipeline(
            LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=seed),
            (1, 1),
        ),
        "logistic_regression_bigram": pipeline(
            LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=seed),
            (1, 2),
        ),
        "logistic_regression_char": pipeline(
            LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=seed),
            (3, 5),
            analyzer="char_wb",
        ),
        "linear_svc": pipeline(
            LinearSVC(class_weight="balanced", random_state=seed), (1, 2)
        ),
    }


def _validated_model(artifact: object) -> Any:
    """Return the model from a compatible trusted artifact."""
    required_keys = {
        "model",
        "model_name",
        "labels",
        "dataset_sha256",
        "seed",
        "task",
        "dataset_domain",
        "provenance",
    }
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise ValueError("Model artifact has an unsupported schema.")
    if artifact["task"] != ARTIFACT_TASK:
        raise ValueError("Model artifact task is not user request classification.")
    if artifact["dataset_domain"] != DATASET_DOMAIN:
        raise ValueError("Model artifact dataset domain is not user requests.")
    if set(map(str, artifact["labels"])) != set(LABELS):
        raise ValueError("Model artifact does not contain the required labels.")

    model = artifact["model"]
    if not hasattr(model, "classes_"):
        raise ValueError("Model artifact contains an unfitted model.")
    if set(map(str, model.classes_)) != set(LABELS):
        raise ValueError("Model classes do not match artifact labels.")
    return model


def classify_request(artifact: object, text: object) -> dict[str, object]:
    """Classify one request and reject text outside the fitted vocabulary."""
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Prediction text must contain non-whitespace characters.")

    model = _validated_model(artifact)
    vectorizer = getattr(model, "named_steps", {}).get("tfidf")
    if vectorizer is None or not hasattr(vectorizer, "transform"):
        raise ValueError("Model artifact does not contain the expected TF-IDF pipeline.")
    features = vectorizer.transform([normalized])
    if getattr(features, "nnz", 0) == 0:
        return {"label": "unclear", "reason": "unknown_terms"}

    label = str(model.predict([normalized])[0])
    if label not in LABELS:
        raise ValueError("Model returned an unsupported label.")

    result: dict[str, object] = {"label": label}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([normalized])[0]
        result["model_probabilities"] = {
            str(model_label): round(float(probability), 6)
            for model_label, probability in zip(model.classes_, probabilities)
        }
    return result
