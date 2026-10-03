"""Text normalization and model definitions for the event classifier."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


_WHITESPACE = re.compile(r"\s+")
_TURKISH_UPPER_I = str.maketrans({"I": "\u0131", "\u0130": "i"})


def normalize_text(value: object) -> str:
    """Normalize Unicode, Turkish uppercase I, case, and whitespace."""
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.translate(_TURKISH_UPPER_I).lower()
    return _WHITESPACE.sub(" ", text).strip()


def make_models(seed: int = 42) -> dict[str, Any]:
    """Create fresh candidate pipelines. Imports stay local for data preparation."""
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    def pipeline(classifier: Any, ngrams: tuple[int, int]) -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=normalize_text,
                        lowercase=False,
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
        "linear_svc": pipeline(
            LinearSVC(class_weight="balanced", random_state=seed), (1, 2)
        ),
    }
