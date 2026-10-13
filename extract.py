"""Turn a request into slots (city, date, event_name) with every saved slot tagger, an optional Jev check and
the follow-ups.

load_models loads the four models of the live demo, all trained on set E: the parser (dictionary baseline) and the
Linear SVM from artifacts/classic, BERTweet from artifacts/bertweet_E and its 6-layer version from
artifacts/bertweet_6L_E_kd0.7. Transformers need torch with a CUDA GPU. When torch is
missing or no GPU is available, transformer artifacts are skipped with a note instead of failing. torch is imported
only inside load_models and load_transformer, so this module imports without it. Spans are rebuilt from the original
text by character offsets, so "chi-town" stays "chi-town". Missing artifacts are skipped and reported in notes.
Without Jev settings the jev field reports not_configured.

Usage: python extract.py [--no-jev] "request text"
"""

import argparse
import json
import sys
import time
from collections import namedtuple
from pathlib import Path

import joblib

from jev import JevClient
from slots import TOKEN, featurize, predict, spans

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
CLASSIC_DIR = ARTIFACTS_DIR / "classic"
SLOT_NAMES = ("city", "date", "event_name")
# Live demo models in display order: the simple parser, the best classic model on dev, and the trained transformer.
# Classic models as (file stem, display name, family); transformers as (artifact dir, display name).
CLASSIC_MODELS = (("dictionary", "Parser", "baseline"), ("linear_svm", "Linear SVM", "classic"))
LIVE_TRANSFORMERS = (("bertweet_E", "BERTweet_E"), ("bertweet_6L_E_kd0.7", "BERTweet_E small (6 of 12 layers)"))
MAIN_MODEL = "bertweet_E"  # its slots drive the follow-up questions and the request log
# Every transformer whose results the Results tab shows.
TRANSFORMER_MODELS = (("bert", "BERT"), ("roberta", "RoBERTa"), ("bertweet", "BERTweet"),
                      ("bertweet_E", "BERTweet_E"))
LoadedModel = namedtuple("LoadedModel", "key name family tag")


def classic_tagger(data):
    """Tag function for a saved classic model: a dictionary tagger, or a vectorizer with an estimator."""
    if "tagger" in data:
        return data["tagger"].tag
    vec, model = data["vectorizer"], data["estimator"]
    return lambda tokens: predict(vec, model, featurize([{"tokens": tokens}]))[0]


def transformer_problem():
    """Why transformers cannot run here, or None when torch and a CUDA GPU are available."""
    try:
        import torch
    except ImportError:
        return "torch is not installed (see requirements-gpu.txt)"
    if not torch.cuda.is_available():
        return "no CUDA GPU is available"
    return None


def load_transformer(path):
    """Tag function for a fine-tuned transformer saved in path. Needs torch and a CUDA GPU."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    import train_bert

    model = AutoModelForTokenClassification.from_pretrained(path).cuda()
    # The saved BERTweet tokenizer is broken, so load the one the model was trained with.
    tokenizer = AutoTokenizer.from_pretrained(model.config.base_model_id)

    def tag(tokens):
        examples = train_bert.encode(tokenizer, [{"tokens": tokens, "tags": ["O"] * len(tokens)}])
        return train_bert.predict(model, examples, tokenizer.pad_token_id)[0]
    return tag


def load_models(classic_dir=CLASSIC_DIR, artifacts_dir=ARTIFACTS_DIR):
    """Every usable slot tagger for the live comparison, in display order, and a note for each skipped model.

    A classic model is skipped when its file is missing. A transformer is skipped when its artifact folder is
    missing, or when transformer_problem() reports that torch or a CUDA GPU is unavailable. Only the first
    transformer that needs torch triggers that check.
    """
    models, notes = [], []
    for key, name, family in CLASSIC_MODELS:
        path = Path(classic_dir) / f"{key}.joblib"
        if path.exists():
            models.append(LoadedModel(key, name, family, classic_tagger(joblib.load(path))))
        else:
            notes.append(f"{name} skipped: {path} not found")
    problem = None
    checked = False
    for key, name in LIVE_TRANSFORMERS:
        path = Path(artifacts_dir) / key
        if not (path / "config.json").exists():
            notes.append(f"{name} skipped: {path} not found")
            continue
        if not checked:
            problem, checked = transformer_problem(), True
        if problem is not None:
            notes.append(f"{name} skipped: {problem}")
            continue
        models.append(LoadedModel(key, name, "transformer", load_transformer(path)))
    return models, notes


def training_vocabulary():
    """Lowercased words of training set E, which every live demo model was trained on."""
    from train_classic import set_e_rows

    return {word.lower() for row in set_e_rows() for word in row["tokens"]}


def new_words(text, vocabulary):
    """Words (not punctuation) of text that never appear in the training data, in order, each once, lowercased."""
    found = []
    for match in TOKEN.finditer(text):
        word = match.group().lower()
        if word[0].isalnum() and word not in vocabulary and word not in found:
            found.append(word)
    return found


def extract_spans(text, tag):
    """Return [{"slot", "start", "end"}] with character offsets into text.

    tag(tokens) gives one BIO tag per TOKEN match in text.
    """
    matches = list(TOKEN.finditer(text))
    tags = tag([m.group() for m in matches]) if matches else []
    return [{"slot": kind, "start": matches[start].start(), "end": matches[end - 1].end()}
            for kind, start, end in sorted(spans(tags), key=lambda span: span[1])]


def group_slots(text, found_spans):
    """Return {slot: [span texts]} for spans from extract_spans."""
    found = {name: [] for name in SLOT_NAMES}
    for span in found_spans:
        found[span["slot"]].append(text[span["start"]:span["end"]])
    return found


def follow_ups(slots, jev_result):
    """Return every question to ask before searching, in order. An empty list means ready to search. Pure function."""
    questions = []
    if not slots["city"]:
        questions.append("ask_city")
    if not slots["date"]:
        questions.append("ask_date")
    unclear = jev_result.get("status") != "ok" or jev_result.get("event_type") == "unknown"
    if not slots["event_name"] and unclear:
        questions.append("ask_what")
    if jev_result.get("status") == "ok" and jev_result.get("tickets") == "unknown":
        questions.append("ask_people")
    return questions


def jev_check(text, jev=None):
    """The Jev answer for text. jev None means Jev was skipped; blank text is never sent to Jev."""
    if jev is None:
        return {"status": "skipped"}
    if not text.strip():
        return {"status": "empty"}
    return jev.classify(text)


def compare(text, models, jev_result):
    """One row per model for text: spans, slots, follow-ups from that model's slots and the shared Jev answer."""
    rows = []
    for model in models:
        started = time.perf_counter()
        found = extract_spans(text, model.tag)
        slots = group_slots(text, found)
        rows.append({"key": model.key, "name": model.name, "family": model.family, "spans": found, "slots": slots,
                     "follow_ups": follow_ups(slots, jev_result),
                     "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", help="the request text")
    parser.add_argument("--no-jev", action="store_true", help="skip the Jev check")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    models, notes = load_models()
    jev_result = jev_check(args.text, None if args.no_jev else JevClient())
    result = {"text": args.text, "jev": jev_result, "models": compare(args.text, models, jev_result), "notes": notes}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
