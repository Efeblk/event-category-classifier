"""Sparse models for classifying a supplied persuasion-technique excerpt."""

from __future__ import annotations

import re
import math
import unicodedata
from collections import Counter
from typing import Any


# Official SemEval-2020 Task 11 names. Commas belong to merged labels.
LABELS = (
    "Appeal_to_Authority",
    "Appeal_to_fear-prejudice",
    "Bandwagon,Reductio_ad_hitlerum",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration,Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Repetition",
    "Slogans",
    "Thought-terminating_Cliches",
    "Whataboutism,Straw_Men,Red_Herring",
)
LABEL_DESCRIPTIONS = {
    "Appeal_to_Authority": "Use an authority's status as support for a claim.",
    "Appeal_to_fear-prejudice": "Appeal to fear or prejudice to persuade.",
    "Bandwagon,Reductio_ad_hitlerum": "Appeal to popularity or discredit through association with a hated group.",
    "Black-and-White_Fallacy": "Present only two choices when other possibilities exist.",
    "Causal_Oversimplification": "Reduce a complex cause to an overly simple explanation.",
    "Doubt": "Question someone's credibility or the reliability of a claim.",
    "Exaggeration,Minimisation": "Overstate or understate the importance of something.",
    "Flag-Waving": "Appeal to national or group identity to justify a claim.",
    "Loaded_Language": "Use emotionally charged wording to influence the reader.",
    "Name_Calling,Labeling": "Use a derogatory or flattering label for a person or group.",
    "Repetition": "Repeat a message to make it more persuasive.",
    "Slogans": "Use a brief, memorable phrase to promote a position.",
    "Thought-terminating_Cliches": "Use a stock phrase to discourage further discussion.",
    "Whataboutism,Straw_Men,Red_Herring": "Deflect attention, distort an opponent's position, or introduce an irrelevant issue.",
}
UNCLEAR = "unclear"  # An out-of-vocabulary abstention, never a training class.
OUTPUT_LABELS = LABELS + (UNCLEAR,)
ARTIFACT_TASK = "persuasion_technique_classification"
DATASET_DOMAIN = "ptc_semeval2020"
ARTIFACT_VERSION = 2

_WHITESPACE = re.compile(r"\s+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"\b[\w']+\b", re.UNICODE)
_MODALS = frozenset(("must", "should", "always", "never", "everyone", "nobody", "all", "only"))


def normalize_text(value: object) -> str:
    """Normalize English Unicode, case and whitespace; retain punctuation."""
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    return _WHITESPACE.sub(" ", text.lower()).strip()


def model_input(text: object, context: object = "") -> dict[str, str]:
    """Keep the original excerpt and optional surrounding text separate."""
    return {"text": "" if text is None else str(text), "context": "" if context is None else str(context)}


def validate_unicode_text(value: str, field: str) -> None:
    """Allow ordinary Unicode and line breaks, but reject invalid/hidden inputs."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field} contains an invalid Unicode character.") from error
    if any(unicodedata.category(character) == "Cc" and character not in "\t\r\n"
           for character in value):
        raise ValueError(f"{field} contains an unsupported control character.")
    if value and not any(not character.isspace() and unicodedata.category(character)[0] not in ("C", "M")
                         for character in value):
        if value.strip():
            raise ValueError(f"{field} must contain visible text.")


def extract_spans(records: list[dict[str, str]]) -> list[str]:
    """Top-level picklable transformer for the excerpt feature channel."""
    return [record["text"] for record in records]


def extract_contexts(records: list[dict[str, str]]) -> list[str]:
    """Top-level picklable transformer for the surrounding-text channel."""
    return [record["context"] for record in records]


def structure_features(records: list[dict[str, str]]) -> list[dict[str, float]]:
    """Twenty-two fixed text-shape features, independent of labels and article IDs."""
    result = []
    for record in records:
        original, excerpt, context = record["text"], normalize_text(record["text"]), normalize_text(record["context"])
        words = _TOKEN.findall(excerpt)
        counts = Counter(words)
        letters = [character for character in original if character.isalpha()]
        n_words, n_chars = max(1, len(words)), max(1, len(original))
        expression = re.compile(r"(?<!\w)" + re.escape(excerpt) + r"(?!\w)")
        repetitions = len(expression.findall(context)) if excerpt else 0
        result.append({
            "log_characters": math.log1p(len(original)), "log_words": math.log1p(len(words)),
            "one_word": float(len(words) == 1), "two_words": float(len(words) == 2),
            "short_phrase": float(3 <= len(words) <= 5), "long_passage": float(len(words) >= 20),
            "mean_word_length": sum(map(len, words)) / n_words,
            "unique_word_fraction": len(counts) / n_words,
            "max_word_repeat_fraction": max(counts.values(), default=0) / n_words,
            "uppercase_fraction": sum(character.isupper() for character in letters) / max(1, len(letters)),
            "digits_fraction": sum(character.isdigit() for character in original) / n_chars,
            "question_mark": float("?" in original), "exclamation_mark": float("!" in original),
            "quote_fraction": sum(original.count(character) for character in ('"', "\u201c", "\u201d")) / n_chars,
            "comma_fraction": original.count(",") / n_chars,
            "semicolon_fraction": original.count(";") / n_chars,
            "colon_fraction": original.count(":") / n_chars,
            "sentence_marks_log": math.log1p(sum(original.count(character) for character in ".!?")),
            "modal_quantifier_fraction": sum(word in _MODALS for word in words) / n_words,
            "negation_fraction": sum(word in {"not", "no", "never", "n't"} for word in words) / n_words,
            "exact_repetitions_log": math.log1p(repetitions),
            "repeated_in_context": float(repetitions > 1),
        })
    return result


def make_models(seed: int = 42, profile: str = "development") -> dict[str, Any]:
    """Create fixed candidates; the historical baseline excludes the upgrade."""
    if profile not in ("development", "baseline"):
        raise ValueError("Model profile must be development or baseline.")
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
    from sklearn.svm import LinearSVC

    def channel(ngrams: tuple[int, int], analyzer: str = "word", context: bool = False) -> Pipeline:
        return Pipeline([
            ("extract", FunctionTransformer(extract_contexts if context else extract_spans, validate=False)),
            ("tfidf", TfidfVectorizer(
                preprocessor=normalize_text, lowercase=False, analyzer=analyzer,
                ngram_range=ngrams, min_df=2, sublinear_tf=True,
            )),
        ])

    def pipeline(classifier: Any, character: bool = False, context: bool = False) -> Pipeline:
        span = channel((3, 5) if character else (1, 2), "char_wb" if character else "word")
        features = FeatureUnion([
            ("span", span), ("context", channel((1, 2), context=True)),
        ]) if context else span
        return Pipeline([("features", features), ("classifier", classifier)])

    def lr() -> LogisticRegression:
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)

    def svm() -> LinearSVC:
        return LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000)

    models = {
        "dummy_most_frequent": pipeline(DummyClassifier(strategy="most_frequent", random_state=seed)),
        "multinomial_nb_word": pipeline(MultinomialNB(alpha=1.0)),
        "multinomial_nb_char": pipeline(MultinomialNB(alpha=1.0), character=True),
        "logistic_regression_word": pipeline(lr()),
        "logistic_regression_char": pipeline(lr(), character=True),
        "linear_svm_word": pipeline(svm()),
        "linear_svm_char": pipeline(svm(), character=True),
        "logistic_regression_word_context": pipeline(lr(), context=True),
        "linear_svm_word_context": pipeline(svm(), context=True),
    }
    if profile == "development":
        features = FeatureUnion([
            ("span_word", channel((1, 2))),
            ("context_word", channel((1, 2), context=True)),
            ("span_char", channel((3, 5), "char_wb")),
            ("structure", Pipeline([
                ("extract", FunctionTransformer(structure_features, validate=False)),
                ("dict", DictVectorizer()), ("scale", MaxAbsScaler()),
            ])),
        ], transformer_weights={"span_word": 1.0, "context_word": 1.0, "span_char": 0.6, "structure": 0.5})
        models["logistic_regression_hybrid_structure"] = Pipeline([("features", features), ("classifier", lr())])
    return models


def _validated_model(artifact: object) -> Any:
    """Validate metadata of a trusted joblib artifact and return its model."""
    required = {
        "schema_version", "model", "model_name", "labels", "dataset_sha256",
        "code_sha256", "seed", "task", "dataset_domain", "provenance",
    }
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Model artifact has an unsupported schema.")
    if artifact["schema_version"] != ARTIFACT_VERSION:
        raise ValueError("Model artifact schema version is unsupported.")
    if artifact["task"] != ARTIFACT_TASK or artifact["dataset_domain"] != DATASET_DOMAIN:
        raise ValueError("Model artifact task or dataset domain is incompatible.")
    if not isinstance(artifact["labels"], (list, tuple)) or tuple(artifact["labels"]) != LABELS:
        raise ValueError("Model artifact does not contain the fixed 14 technique labels.")
    for key in ("dataset_sha256", "code_sha256"):
        if not isinstance(artifact[key], str) or not _SHA256.fullmatch(artifact[key]):
            raise ValueError(f"Model artifact {key} must be a SHA256 digest.")
    if not isinstance(artifact["seed"], int) or isinstance(artifact["seed"], bool):
        raise ValueError("Model artifact seed must be an integer.")
    if not isinstance(artifact["model_name"], str) or not artifact["model_name"]:
        raise ValueError("Model artifact needs a model name.")
    provenance = artifact["provenance"]
    if not isinstance(provenance, dict) or provenance.get("fit_partition") != "train":
        raise ValueError("Model artifact must identify train-only provenance.")
    scopes = {"dev_only": 1000, "historical_baseline": 350}
    scope = provenance.get("evaluation_scope")
    if not isinstance(scope, str) or scope not in scopes or provenance.get("context_margin") != scopes[scope]:
        raise ValueError("Model artifact needs a compatible evaluation scope and context margin.")
    model = artifact["model"]
    if not hasattr(model, "classes_") or set(map(str, model.classes_)) != set(LABELS):
        raise ValueError("Fitted model classes do not match the required labels.")
    if "features" not in getattr(model, "named_steps", {}) or not hasattr(model, "predict"):
        raise ValueError("Model artifact lacks the expected feature pipeline.")
    return model


def _feature_names(features: Any) -> list[str]:
    """Name both feature channels without requiring extractor column names."""
    if hasattr(features, "transformer_list"):
        return [f"{channel}: {name}" for channel, pipeline in features.transformer_list
                for name in pipeline.named_steps["tfidf" if "tfidf" in pipeline.named_steps else "dict"].get_feature_names_out()]
    return [f"span: {name}" for name in features.named_steps["tfidf"].get_feature_names_out()]


def _has_excerpt_features(transformer: Any, matrix: Any) -> bool:
    """Context and numeric shape cannot substitute for known excerpt terms."""
    if not hasattr(transformer, "transformer_list"):
        return matrix.nnz > 0
    offset = 0
    for name, channel in transformer.transformer_list:
        lexical = "tfidf" in channel.named_steps
        vectorizer = channel.named_steps["tfidf" if lexical else "dict"]
        count = len(vectorizer.get_feature_names_out())
        if lexical and name in ("span", "span_word", "span_char") and matrix[:, offset:offset + count].nnz:
            return True
        offset += count
    return False


def _top_features(model: Any, features: Any, label: str) -> list[dict[str, object]]:
    """Return positive feature contributions, not causal explanations."""
    classifier = model.named_steps["classifier"]
    class_index = list(map(str, model.classes_)).index(label)
    if hasattr(classifier, "coef_"):
        weights = classifier.coef_[class_index]
    elif hasattr(classifier, "feature_log_prob_"):
        weights = classifier.feature_log_prob_[class_index] - classifier.feature_log_prob_.mean(axis=0)
    else:
        return []
    names = _feature_names(model.named_steps["features"])
    contributions = [(int(index), float(value * weights[index]))
                     for index, value in zip(features.indices, features.data)]
    positive = sorted((pair for pair in contributions if pair[1] > 0), key=lambda pair: (-pair[1], pair[0]))
    return [{"feature": names[index], "weight": round(weight, 6)} for index, weight in positive[:8]]


def classify_request(artifact: object, text: object, context: object = "") -> dict[str, object]:
    """Classify a supplied excerpt; this does not locate spans or verify claims."""
    if not isinstance(text, str) or not normalize_text(text):
        raise ValueError("Prediction excerpt must contain non-whitespace characters.")
    if not isinstance(context, str):
        raise ValueError("Prediction context must be text.")
    validate_unicode_text(text, "Excerpt")
    validate_unicode_text(context, "Context")
    model = _validated_model(artifact)
    record = model_input(text, context)
    features = model.named_steps["features"].transform([record])
    if not _has_excerpt_features(model.named_steps["features"], features):
        return {"label": UNCLEAR, "reason": "unknown_terms"}
    label = str(model.predict([record])[0])
    if label not in LABELS:
        raise ValueError("Model returned an unsupported label.")
    result: dict[str, object] = {"label": label, "top_features": _top_features(model, features, label)}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([record])[0]
        result["model_probabilities"] = {
            str(model_label): round(float(probability), 6)
            for model_label, probability in zip(model.classes_, probabilities)
        }
    return result


classify_excerpt = classify_request
