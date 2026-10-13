"""Shared helpers for slot tagging: JSONL and BIO tags, token features, typo noise, cleaning, span scoring and the
dictionary tagger."""

import json
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction import DictVectorizer

BOUNDARY = "<BOUNDARY>"
TOKEN = re.compile(r"\w+|[^\w\s]")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().split("\n") if line.strip()]


def write_lines(path, lines):
    """Write lines as UTF-8 with LF endings, one newline after each line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(line + "\n" for line in lines)


def bio_tags(text, spans):
    """Tag each token with B-/I-/O from (start, end, slot) spans. Return None on overlapping spans."""
    tokens, tags, started = [], [], set()
    for match in re.finditer(TOKEN, text):
        hits = [s for s in spans if match.start() < s[1] and match.end() > s[0]]
        if len(hits) > 1:
            return None
        tokens.append(match.group())
        if not hits:
            tags.append("O")
            continue
        span = hits[0]
        tags.append(f"{'I' if span in started else 'B'}-{span[2]}")
        started.add(span)
    return tokens, tags


def dedupe(rows):
    """Keep the first row for each lowercased text; return (kept, number dropped)."""
    seen, kept = set(), []
    for row in rows:
        key = row["text"].lower()
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept, len(rows) - len(kept)


def drop_leaks(rows, banned):
    """Drop rows whose lowercased text is in banned; return (kept, number dropped)."""
    kept = [r for r in rows if r["text"].lower() not in banned]
    return kept, len(rows) - len(kept)


def clean_split(rows, earlier):
    """Dedupe one split, then drop texts already kept in the earlier splits (train, then dev).

    Return (kept, duplicates dropped, leaked rows dropped). The first split passes earlier=[].
    """
    rows, duplicates = dedupe(rows)
    if not earlier:
        return rows, duplicates, 0
    rows, leakage = drop_leaks(rows, {r["text"].lower() for split in earlier for r in split})
    return rows, duplicates, leakage


def slot_counts(rows):
    """Number of spans per slot type in a split, sorted by slot name."""
    return dict(sorted(Counter(tag[2:] for r in rows for tag in r["tags"] if tag.startswith("B-")).items()))


def describe(tokens, tags):
    return "; ".join(f"{kind}: {' '.join(tokens[s:e])}" for kind, s, e in sorted(spans(tags), key=lambda x: x[1]))


def word_shape(word):
    """Map letters to X/x and digits to d, collapsing runs: 'Yankees' -> 'Xx', '2026' -> 'd'."""
    chars = ["X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c for c in word]
    return "".join(c for i, c in enumerate(chars) if i == 0 or c != chars[i - 1] or c not in "Xxd")


def token_features(tokens, i):
    """Feature dict for tokens[i]: word, affixes, shape, capitalisation and neighbouring words."""
    word = tokens[i]
    lower = word.lower()
    n = len(tokens)
    feats = {
        "word": lower,
        "is_title": word.istitle(),
        "is_upper": word.isupper(),
        "has_digit": any(c.isdigit() for c in word),
        "is_first": i == 0,
        "is_last": i == n - 1,
        "shape": word_shape(word),
    }
    for size in (2, 3, 4):
        feats[f"prefix{size}"] = lower[:size]
        feats[f"suffix{size}"] = lower[-size:]
    for offset in (-2, -1, 1, 2):
        j = i + offset
        feats[f"word{offset:+d}"] = tokens[j].lower() if 0 <= j < n else BOUNDARY
    for offset in (-1, 1):
        j = i + offset
        feats[f"is_title{offset:+d}"] = tokens[j].istitle() if 0 <= j < n else False
    return feats


def featurize(rows):
    """Token feature dicts for every token of every row, one list per row."""
    return [[token_features(r["tokens"], i) for i in range(len(r["tokens"]))] for r in rows]


def add_typos(tokens, rng, rate):
    """Return a noisy copy of tokens: each letter-only token of length >= 4 gets one edit with probability rate.

    The edit deletes a letter, doubles a letter, or swaps two neighbouring letters that do not include the first
    letter. Tags are unchanged, so the caller keeps the original tag sequence.
    """
    noisy = []
    for word in tokens:
        if len(word) >= 4 and word.isalpha() and rng.random() < rate:
            op = rng.choice(["delete", "double", "swap"])
            if op == "delete":
                i = rng.randrange(len(word))
                word = word[:i] + word[i + 1:]
            elif op == "double":
                i = rng.randrange(len(word))
                word = word[:i + 1] + word[i] + word[i + 1:]
            else:
                i = rng.randrange(1, len(word) - 1)
                word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        noisy.append(word)
    return noisy


SLANG = {"tomorrow": "tmrw", "tonight": "tonite", "today": "2day", "weekend": "wknd", "next": "nxt", "you": "u",
         "are": "r", "for": "4", "please": "pls", "with": "w", "what": "wat", "going": "goin", "something": "smth",
         "people": "ppl", "thanks": "thx"}


def messy_copy(tokens, rng, slang_rate=0.5, typo_rate=0.15):
    """Return a chat-style copy of tokens: lowercased, common words swapped for slang (SLANG) with probability
    slang_rate, then typo noise from add_typos. The token count stays the same, so the original tags still apply.
    """
    words = [w.lower() for w in tokens]
    words = [SLANG[w] if w in SLANG and rng.random() < slang_rate else w for w in words]
    return add_typos(words, rng, typo_rate)


def repair_bio(tags):
    """Turn an I-x that follows O or a different type into B-x."""
    fixed = []
    for tag in tags:
        if tag.startswith("I-"):
            kind = tag[2:]
            previous = fixed[-1] if fixed else "O"
            if previous == "O" or previous[2:] != kind:
                tag = "B-" + kind
        fixed.append(tag)
    return fixed


def spans(tags):
    """Return the set of (type, start, end) spans in a BIO tag sequence, end exclusive."""
    found = set()
    current = None
    for i, tag in enumerate(repair_bio(tags) + ["O"]):
        if tag == "O" or tag.startswith("B-"):
            if current is not None:
                found.add((current[0], current[1], i))
            current = (tag[2:], i) if tag != "O" else None
    return found


def slot_f1(scores, slot):
    """Test F1 of one slot as a 3-decimal string, or "-" when the slot never occurs."""
    return f"{scores['slots'][slot]['f1']:.3f}" if slot in scores["slots"] else "-"


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def span_scores(gold_list, pred_list):
    """Micro P/R/F1 and per-slot P/R/F1/support over exact spans, plus the share of sentences matched exactly."""
    counts = defaultdict(lambda: [0, 0, 0])  # slot -> [tp, fp, fn]
    exact = 0
    for gold_tags, pred_tags in zip(gold_list, pred_list):
        gold, pred = spans(gold_tags), spans(pred_tags)
        exact += gold == pred
        for span in gold & pred:
            counts[span[0]][0] += 1
        for span in pred - gold:
            counts[span[0]][1] += 1
        for span in gold - pred:
            counts[span[0]][2] += 1
    slots = {slot: {**prf(*c), "support": c[0] + c[2]} for slot, c in sorted(counts.items())}
    totals = [sum(c[i] for c in counts.values()) for i in range(3)]
    return {
        **prf(*totals),
        "sentence_exact": exact / len(gold_list) if gold_list else 0.0,
        "slots": slots,
    }


def span_counts(gold_list, pred_list):
    """Exact-span true positives, false positives and false negatives of each sentence, as an (n, 3) array."""
    rows = []
    for gold_tags, pred_tags in zip(gold_list, pred_list):
        gold, pred = spans(gold_tags), spans(pred_tags)
        rows.append((len(gold & pred), len(pred - gold), len(gold - pred)))
    return np.array(rows, dtype=float).reshape(-1, 3)


def bootstrap_f1(gold_list, pred_list, base_list=None, rounds=1000, seed=0):
    """95% bootstrap interval of micro span F1: draw the test sentences again with replacement, `rounds` times, and
    keep the middle 95% of the F1 scores. With base_list, the interval is of F1(pred) - F1(base) on the same draws
    (paired). Returns (low, high)."""
    def f1(counts):
        tp, fp, fn = counts.sum(axis=1).T
        return np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=2 * tp + fp + fn > 0)

    picks = np.random.default_rng(seed).integers(0, len(gold_list), size=(rounds, len(gold_list)))
    scores = f1(span_counts(gold_list, pred_list)[picks])
    if base_list is not None:
        scores = scores - f1(span_counts(gold_list, base_list)[picks])
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


CONFUSION_LABELS = ("city", "date", "event_name", "O")


def word_confusion(gold_list, pred_list):
    """Word-level confusion matrix with B-/I- dropped: {gold slot: {predicted slot: words}}, "O" for no slot."""
    matrix = {gold: dict.fromkeys(CONFUSION_LABELS, 0) for gold in CONFUSION_LABELS}
    for gold_tags, pred_tags in zip(gold_list, pred_list):
        for gold, pred in zip(gold_tags, repair_bio(pred_tags)):
            matrix[gold[2:] or "O"][pred[2:] or "O"] += 1
    return matrix


def to_int32(X):
    """LinearSVC rejects int64 sparse indices, which DictVectorizer produces."""
    X = X.tocsr()
    X.indices = X.indices.astype(np.int32)
    X.indptr = X.indptr.astype(np.int32)
    return X


def fit_vectorizer(feats):
    """Fit a DictVectorizer on flat per-token features and return (vectorizer, matrix)."""
    vec = DictVectorizer()
    return vec, to_int32(vec.fit_transform([x for seq in feats for x in seq]))


def predict(vec, model, feats):
    """BIO tags per sentence from a fitted vectorizer and estimator; one list of tags per sentence."""
    flat = [x for seq in feats for x in seq]
    labels = [str(x) for x in model.predict(to_int32(vec.transform(flat)))]
    out, pos = [], 0
    for sent in feats:
        out.append(labels[pos:pos + len(sent)])
        pos += len(sent)
    return out


class DictionaryTagger:
    """Memorise each span text -> its most common slot type; tag longest matches left to right."""

    def __init__(self, rows):
        votes = defaultdict(Counter)
        for r in rows:
            for kind, start, end in spans(r["tags"]):
                votes[tuple(w.lower() for w in r["tokens"][start:end])][kind] += 1
        self.best = {key: c.most_common(1)[0][0] for key, c in votes.items()}
        self.lengths = sorted({len(k) for k in self.best}, reverse=True)

    def tag(self, tokens):
        words = [t.lower() for t in tokens]
        tags = ["O"] * len(words)
        i = 0
        while i < len(words):
            for n in self.lengths:
                kind = self.best.get(tuple(words[i:i + n])) if i + n <= len(words) else None
                if kind:
                    tags[i:i + n] = [f"B-{kind}"] + [f"I-{kind}"] * (n - 1)
                    i += n
                    break
            else:
                i += 1
        return tags
