"""Optional whitespace-token slot classifiers and exact span metrics."""

from __future__ import annotations

import re
from collections import Counter

from classifier import normalize_text
from parser import ascii_text


HIGHLIGHT_SLOTS = ("date", "time", "timeofday", "place_name", "event_name", "artist_name", "person")
PARSER_SLOT_TYPES = ("date", "time", "timeofday")


def token_features(tokens: list[str], index: int) -> dict:
    """Describe one token by itself, its prefixes and suffixes, and its neighbours."""
    word = normalize_text(tokens[index])
    features = {
        "word": word,
        "number": any(char.isdigit() for char in word),
        "capital": tokens[index][:1].isupper(),
        "apostrophe": "'" in word or "’" in word,
        "start": index == 0,
        "end": index == len(tokens) - 1,
    }
    for size in (3, 4, 5):
        features[f"prefix{size}"] = word[:size]
    # Turkish suffixes carry case and tense, so word endings are strong features.
    for size in (2, 3, 4):
        features[f"suffix{size}"] = word[-size:]
    for offset in (-2, -1, 1, 2):
        neighbor = index + offset
        if 0 <= neighbor < len(tokens):
            features[f"word{offset:+}"] = normalize_text(tokens[neighbor])
        else:
            features[f"word{offset:+}"] = "<BOUNDARY>"
    return features


def repair_bio(tags: list[str]) -> list[str]:
    """An I-type after O or after a different type starts a new B-type span."""
    repaired = []
    previous = None
    for tag in tags:
        if tag == "O":
            previous = None
        else:
            if not re.fullmatch(r"[BI]-[a-z_]+", tag):
                raise ValueError("Invalid BIO tag.")
            kind = tag[2:]
            if tag.startswith("I-") and previous != kind:
                tag = "B-" + kind
            previous = kind
        repaired.append(tag)
    return repaired


def spans(tags: list[str]) -> set[tuple[str, int, int]]:
    """Return (type, start token, exclusive end token) for each span."""
    result = set()
    start, kind = None, None
    for index, tag in enumerate(repair_bio(tags) + ["O"]):
        if tag == "O" or tag.startswith("B-"):
            if kind is not None:
                result.add((kind, start, index))
            if tag == "O":
                start, kind = None, None
            else:
                start, kind = index, tag[2:]
    return result


def _prf(true_positive: int, predicted: int, gold: int) -> dict:
    return {
        "precision": true_positive / predicted if predicted else 0.0,
        "recall": true_positive / gold if gold else 0.0,
        "f1": 2 * true_positive / (predicted + gold) if predicted + gold else 0.0,
        "true_positive": true_positive,
        "predicted": predicted,
        "gold": gold,
    }


def slot_scores(actual: list[list[str]], predicted: list[list[str]], allowed_types=None) -> dict:
    """Micro span F1: a span counts only with the exact type and token boundaries."""
    if len(actual) != len(predicted):
        raise ValueError("Different sentence counts.")
    gold, guesses, correct = Counter(), Counter(), Counter()
    errors = Counter()
    for truth, guess in zip(actual, predicted):
        if len(truth) != len(guess):
            raise ValueError("Different token counts.")
        gold_spans, guess_spans = spans(truth), spans(guess)
        if allowed_types is not None:
            gold_spans = {span for span in gold_spans if span[0] in allowed_types}
            guess_spans = {span for span in guess_spans if span[0] in allowed_types}
        gold.update(span[0] for span in gold_spans)
        guesses.update(span[0] for span in guess_spans)
        correct.update(span[0] for span in gold_spans & guess_spans)
        errors.update(classify_errors(gold_spans - guess_spans, guess_spans - gold_spans))

    kinds = sorted(set(gold) | set(guesses))
    return {
        **_prf(sum(correct.values()), sum(guesses.values()), sum(gold.values())),
        "per_slot": {kind: _prf(correct[kind], guesses[kind], gold[kind]) for kind in kinds},
        "errors": {key: errors[key] for key in ("boundary", "type_confusion", "missed", "spurious")},
    }


def classify_errors(missed: set, spurious: set) -> Counter:
    """Pair wrong spans: same boundaries means a type error, same type and overlap a boundary error."""
    missed, spurious = set(missed), set(spurious)
    counts = Counter()
    for gold in sorted(missed):
        same_boundaries = sorted(guess for guess in spurious if gold[1:] == guess[1:])
        if same_boundaries:
            counts["type_confusion"] += 1
            missed.remove(gold)
            spurious.remove(same_boundaries[0])
    for gold in sorted(missed):
        overlapping = sorted(
            guess for guess in spurious
            if gold[0] == guess[0] and max(gold[1], guess[1]) < min(gold[2], guess[2])
        )
        if overlapping:
            counts["boundary"] += 1
            missed.remove(gold)
            spurious.remove(overlapping[0])
    counts["missed"] += len(missed)
    counts["spurious"] += len(spurious)
    return counts


def exact_match(actual_intents, predicted_intents, actual_tags, predicted_tags) -> float:
    """Share of sentences with the right intent and exactly the right slots."""
    lengths = {len(values) for values in (actual_intents, predicted_intents, actual_tags, predicted_tags)}
    if len(lengths) != 1:
        raise ValueError("Different sentence counts.")
    if not actual_intents:
        return 0.0
    matches = sum(
        gold_intent == intent and spans(gold) == spans(guess)
        for gold_intent, intent, gold, guess in zip(actual_intents, predicted_intents, actual_tags, predicted_tags)
    )
    return matches / len(actual_intents)


def make_slot_models(seed: int = 42) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB
    from sklearn.svm import LinearSVC

    return {
        "all_O": None,  # majority baseline: every token is outside a slot
        "multinomial_nb": MultinomialNB(),
        "bernoulli_nb": BernoulliNB(),
        "logistic_regression": LogisticRegression(max_iter=300, random_state=seed),
        "linear_svm": LinearSVC(random_state=seed),
    }


def compact_indices(matrix):
    """LinearSVC needs 32-bit sparse indices; our matrix is far below that limit."""
    if max(matrix.shape, default=0) >= 2**31 or matrix.nnz >= 2**31:
        raise ValueError("Slot feature matrix exceeds 32-bit sparse index limits.")
    matrix.indices = matrix.indices.astype("int32")
    matrix.indptr = matrix.indptr.astype("int32")
    return matrix


def predict_tags(artifact: dict, sentences: list[list[str]]) -> list[list[str]]:
    """Tag every token of every sentence, then repair invalid BIO sequences."""
    if (
        not isinstance(artifact, dict)
        or artifact.get("task") != "slot_token_classification"
        or artifact.get("dataset_domain") != "massive_tr"
    ):
        raise ValueError("Incompatible slot artifact.")
    if not {"vectorizer", "model", "tags"}.issubset(artifact):
        raise ValueError("Incomplete slot artifact.")
    model = artifact["model"]
    if model is not None and set(map(str, model.classes_)) != set(artifact["tags"]):
        raise ValueError("Slot artifact tags do not match model.")

    features = [token_features(tokens, i) for tokens in sentences for i in range(len(tokens))]
    if not features:
        return [[] for _ in sentences]
    if model is None:
        predictions = ["O"] * len(features)
    else:
        matrix = compact_indices(artifact["vectorizer"].transform(features))
        predictions = [str(tag) for tag in model.predict(matrix)]

    result = []
    offset = 0
    for tokens in sentences:
        result.append(repair_bio(predictions[offset:offset + len(tokens)]))
        offset += len(tokens)
    return result


# Regex rules written from train only. They cover just three slot types.
DATE = r"(?:bu\s+(?:hafta|yil|ay)|(?:bugun|yarin|pazartesi|sali|carsamba|persembe|cuma|cumartesi|pazar)[a-z']*)"
CLOCK = (
    r"(?:\d+(?::\d+)?(?:'(?:te|ta|de|da))?"
    r"|birde|ikide|ucte|dortte|beste|altida|yedide|sekizde|dokuzda|onda"
    r"|bire|ikiye|uce|dorde|bese|altiya|yediye|sekize|dokuza|ona)"
)
TIMEOFDAY = r"(?:bu\s+gece|ogleden\s+sonra|sabah|aksam|gece|oglen)"


def parser_tags(tokens: list[str]) -> list[str]:
    words = [ascii_text(token) for token in tokens]
    text = " ".join(words)
    boundaries = []
    offset = 0
    for word in words:
        boundaries.append((offset, offset + len(word)))
        offset += len(word) + 1

    tags = ["O"] * len(tokens)
    # Times go first, so a daypart inside a time ("aksam dokuzda") stays part of it.
    time_pattern = rf"(?:{TIMEOFDAY}\s+)?{CLOCK}"
    for kind, pattern in (("time", time_pattern), ("date", DATE), ("timeofday", TIMEOFDAY)):
        for match in re.finditer(rf"(?<!\S){pattern}(?!\S)", text):
            indices = [i for i, (start, end) in enumerate(boundaries) if match.start() <= start and end <= match.end()]
            if indices and all(tags[i] == "O" for i in indices):
                for n, i in enumerate(indices):
                    tags[i] = ("B-" if n == 0 else "I-") + kind
    return tags


def display_spans(text: str, tags: list[str]) -> list[dict]:
    """Map token spans back to character offsets in the original text."""
    tokens = list(re.finditer(r"\S+", text))
    if len(tokens) != len(tags):
        raise ValueError("Different token counts.")
    return [
        {
            "type": kind,
            "start": tokens[start].start(),
            "end": tokens[end - 1].end(),
            "text": text[tokens[start].start():tokens[end - 1].end()],
        }
        for kind, start, end in sorted(spans(tags), key=lambda span: span[1])
    ]
