"""Optional whitespace-token slot classifiers and exact span metrics."""
from __future__ import annotations
import re
from collections import Counter
from classifier import normalize_text
from parser import ascii_text

HIGHLIGHT_SLOTS = ("date", "time", "timeofday", "place_name", "event_name", "artist_name", "person")
PARSER_SLOT_TYPES = ("date", "time", "timeofday")


def token_features(tokens: list[str], index: int) -> dict:
    word = normalize_text(tokens[index])
    features = {"word": word, "number": any(c.isdigit() for c in word),
                "capital": tokens[index][:1].isupper(), "apostrophe": "'" in word or "’" in word,
                "start": index == 0, "end": index == len(tokens)-1}
    for size in (3,4,5): features[f"prefix{size}"] = word[:size]
    for size in (2,3,4): features[f"suffix{size}"] = word[-size:]
    for offset in (-2,-1,1,2):
        neighbor = index + offset
        features[f"word{offset:+}"] = normalize_text(tokens[neighbor]) if 0 <= neighbor < len(tokens) else "<BOUNDARY>"
    return features


def repair_bio(tags: list[str]) -> list[str]:
    """An I-type after O or a different type starts a new B-type span."""
    repaired = []
    previous = None
    for tag in tags:
        if tag == "O": previous = None
        else:
            if not re.fullmatch(r"[BI]-[a-z_]+", tag): raise ValueError("Invalid BIO tag.")
            kind = tag[2:]
            if tag.startswith("I-") and previous != kind: tag = "B-" + kind
            previous = kind
        repaired.append(tag)
    return repaired


def spans(tags: list[str]) -> set[tuple[str,int,int]]:
    """Return (type, start token, exclusive end token)."""
    result = set()
    start, kind = None, None
    for index, tag in enumerate(repair_bio(tags) + ["O"]):
        if tag == "O" or tag.startswith("B-"):
            if kind is not None: result.add((kind,start,index))
            start, kind = (None,None) if tag == "O" else (index,tag[2:])
    return result


def _prf(tp, predicted, gold):
    precision = tp/predicted if predicted else 0.0
    recall = tp/gold if gold else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2*tp/(predicted+gold) if predicted+gold else 0.0,
            "true_positive": tp, "predicted": predicted, "gold": gold}


def slot_scores(actual, predicted, allowed_types=None):
    if len(actual) != len(predicted): raise ValueError("Different sentence counts.")
    gold, guesses, correct = Counter(),Counter(),Counter()
    boundaries = types = missing = extra = 0
    for truth, guess in zip(actual,predicted):
        if len(truth) != len(guess): raise ValueError("Different token counts.")
        a,b = spans(truth),spans(guess)
        if allowed_types is not None:
            a = {s for s in a if s[0] in allowed_types}; b = {s for s in b if s[0] in allowed_types}
        gold.update(s[0] for s in a); guesses.update(s[0] for s in b); correct.update(s[0] for s in a & b)
        # Pair exact-boundary type errors first, then same-type overlapping boundaries.
        remaining_a,remaining_b = set(a-b),set(b-a)
        for left in sorted(remaining_a):
            candidates = sorted(right for right in remaining_b if left[1:] == right[1:])
            if candidates:
                types += 1;remaining_a.remove(left);remaining_b.remove(candidates[0])
        for left in sorted(remaining_a):
            candidates = sorted(right for right in remaining_b if left[0] == right[0] and max(left[1],right[1]) < min(left[2],right[2]))
            if candidates:
                boundaries += 1;remaining_a.remove(left);remaining_b.remove(candidates[0])
        missing += len(remaining_a);extra += len(remaining_b)
    kinds = sorted(set(gold)|set(guesses))
    return {**_prf(sum(correct.values()),sum(guesses.values()),sum(gold.values())),
            "per_slot": {k:_prf(correct[k],guesses[k],gold[k]) for k in kinds},
            "errors": {"boundary":boundaries,"type_confusion":types,"missed":missing,"spurious":extra}}


def exact_match(actual_intents, predicted_intents, actual_tags, predicted_tags):
    lengths = {len(x) for x in (actual_intents,predicted_intents,actual_tags,predicted_tags)}
    if len(lengths) != 1: raise ValueError("Different sentence counts.")
    return sum(a == b and spans(c) == spans(d) for a,b,c,d in zip(actual_intents,predicted_intents,actual_tags,predicted_tags))/len(actual_intents) if actual_intents else 0.0


def make_slot_models(seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    from sklearn.svm import LinearSVC
    return {"all_O": None, "multinomial_nb":MultinomialNB(), "bernoulli_nb":BernoulliNB(),
            "logistic_regression":LogisticRegression(max_iter=300,random_state=seed),
            "linear_svm":LinearSVC(random_state=seed)}


def predict_tags(artifact, sentences):
    if not isinstance(artifact,dict) or artifact.get("task") != "slot_token_classification" or artifact.get("dataset_domain") != "massive_tr":
        raise ValueError("Incompatible slot artifact.")
    if not {"vectorizer","model","tags"}.issubset(artifact): raise ValueError("Incomplete slot artifact.")
    model = artifact["model"]
    if model is not None and set(map(str,model.classes_)) != set(artifact["tags"]): raise ValueError("Slot artifact tags do not match model.")
    features = [token_features(tokens,i) for tokens in sentences for i in range(len(tokens))]
    if not features: return [[] for _ in sentences]
    predictions = ["O"]*len(features) if model is None else list(map(str,model.predict(artifact["vectorizer"].transform(features))))
    result = [];offset=0
    for tokens in sentences:
        result.append(repair_bio(predictions[offset:offset+len(tokens)]));offset+=len(tokens)
    return result


# Regex rules from train only; intentionally cover just three slot types.
DATE = r"(?:bu\s+(?:hafta|yil|ay)|(?:bugun|yarin|pazartesi|sali|carsamba|persembe|cuma|cumartesi|pazar)[a-z']*)"
CLOCK = r"(?:\d+(?::\d+)?(?:'(?:te|ta|de|da))?|birde|ikide|ucte|dortte|beste|altida|yedide|sekizde|dokuzda|onda|bire|ikiye|uce|dorde|bese|altiya|yediye|sekize|dokuza|ona)"
TIMEOFDAY = r"(?:bu\s+gece|ogleden\s+sonra|sabah|aksam|gece|oglen)"


def parser_tags(tokens):
    words = [ascii_text(token) for token in tokens]
    text = " ".join(words)
    boundaries = [];offset=0
    for word in words:
        boundaries.append((offset,offset+len(word)));offset+=len(word)+1
    tags = ["O"]*len(tokens)
    # Times first so a daypart included in time is not overwritten.
    time_pattern = rf"(?:{TIMEOFDAY}\s+)?{CLOCK}"
    for kind, pattern in (("time",time_pattern),("date",DATE),("timeofday",TIMEOFDAY)):
        for match in re.finditer(rf"(?<!\S){pattern}(?!\S)",text):
            indices = [i for i,(a,b) in enumerate(boundaries) if match.start() <= a and b <= match.end()]
            if indices and all(tags[i] == "O" for i in indices):
                for n,i in enumerate(indices): tags[i] = ("B-" if n==0 else "I-")+kind
    return tags


def display_spans(text, tags):
    tokens = list(re.finditer(r"\S+",text))
    if len(tokens) != len(tags): raise ValueError("Different token counts.")
    return [{"type":kind,"start":tokens[a].start(),"end":tokens[b-1].end(),
             "text":text[tokens[a].start():tokens[b-1].end()]}
            for kind,a,b in sorted(spans(tags),key=lambda s:s[1])]
