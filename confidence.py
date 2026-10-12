"""95% bootstrap confidence intervals for the main test scores, and for the main differences between models.

The test sets are small (50 messy requests, 260 Codex requests), so one F1 number hides how much it could move with
other requests. For each model and test set this draws the test sentences again with replacement 1,000 times, scores
each draw with micro exact-span F1 and keeps the middle 95% (slots.bootstrap_f1). Comparisons are paired: both models
are scored on the same draws, and a difference whose interval stays above 0 is not explained by the choice of test
requests alone. No retraining and no new data.

Models: the four classic models trained on set E (artifacts/classic) and the saved transformers (artifacts/). Writes
results/confidence_intervals.json. Needs torch with a CUDA GPU for the transformers.

Usage: python confidence.py
"""

import json
from pathlib import Path

import joblib

from extract import classic_tagger
from slots import bootstrap_f1, read_jsonl, span_scores

ROOT = Path(__file__).resolve().parent
ROUNDS = 1000
CLASSIC = (("dictionary", "Parser (set E)"), ("naive_bayes", "Naive Bayes (set E)"),
           ("logistic_regression", "Logistic Regression (set E)"), ("linear_svm", "Linear SVM (set E)"))
TRANSFORMERS = (("bert", "BERT (set D)"), ("roberta", "RoBERTa (set D)"), ("bertweet", "BERTweet (set D)"),
                ("bertweet_E", "BERTweet_E"), ("bertweet_6L_E_kd0.7", "BERTweet_E 6 layers"))
# (model, baseline): the claims the README makes about these pairs.
PAIRS = (("BERTweet_E", "BERTweet (set D)"), ("BERTweet_E", "Linear SVM (set E)"),
         ("BERTweet_E", "RoBERTa (set D)"), ("BERTweet_E 6 layers", "BERTweet_E"))


def test_sets():
    codex = read_jsonl(ROOT / "data" / "codex" / "codex_test.jsonl")
    return {"SGD test": read_jsonl(ROOT / "data" / "clean" / "test.jsonl"),
            "messy": read_jsonl(ROOT / "data" / "messy" / "messy_test.jsonl"),
            "codex_all": codex + read_jsonl(ROOT / "data" / "codex" / "codex_more_test.jsonl")}


def transformer_predictions(path, tests):
    """Tags for every test set, in batches as train_bert scores them (one request at a time can differ a little in
    bf16), so the F1 matches results/<tag>_results.json."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    import train_bert

    model = AutoModelForTokenClassification.from_pretrained(path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model.config.base_model_id)
    return {test: train_bert.predict(model, train_bert.encode(tokenizer, rows), tokenizer.pad_token_id)
            for test, rows in tests.items()}


def main():
    tests = test_sets()
    preds = {}
    for key, name in CLASSIC:
        tag = classic_tagger(joblib.load(ROOT / "artifacts" / "classic" / f"{key}.joblib"))
        preds[name] = {test: [tag(r["tokens"]) for r in rows] for test, rows in tests.items()}
    for key, name in TRANSFORMERS:
        preds[name] = transformer_predictions(ROOT / "artifacts" / key, tests)
    out = {"rounds": ROUNDS, "sentences": {test: len(rows) for test, rows in tests.items()},
           "models": {}, "differences": {}}
    for name, by_test in preds.items():
        out["models"][name] = {}
        for test, rows in tests.items():
            gold = [r["tags"] for r in rows]
            low, high = bootstrap_f1(gold, by_test[test], rounds=ROUNDS)
            out["models"][name][test] = {"f1": span_scores(gold, by_test[test])["f1"], "low": low, "high": high}
    for model, base in PAIRS:
        key = f"{model} - {base}"
        out["differences"][key] = {}
        for test, rows in tests.items():
            gold = [r["tags"] for r in rows]
            low, high = bootstrap_f1(gold, preds[model][test], base_list=preds[base][test], rounds=ROUNDS)
            diff = span_scores(gold, preds[model][test])["f1"] - span_scores(gold, preds[base][test])["f1"]
            out["differences"][key][test] = {"diff": diff, "low": low, "high": high}

    path = ROOT / "results" / "confidence_intervals.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'model':<30}" + "".join(f"{test:>24}" for test in tests))
    for name, by_test in out["models"].items():
        print(f"{name:<30}" + "".join(f"{s['f1']:>8.3f} [{s['low']:.3f}, {s['high']:.3f}]" for s in by_test.values()))
    print()
    for key, by_test in out["differences"].items():
        print(f"{key:<45}" + "".join(f"{s['diff']:>+8.3f} [{s['low']:+.3f}, {s['high']:+.3f}]"
                                     for s in by_test.values()))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
