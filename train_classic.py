"""Train the classic slot taggers: dictionary baseline, Naive Bayes, Logistic Regression and Linear SVM.

Four training sets, labelled A to D in the output: A SGD, B SGD+MASSIVE, C SGD+typo copies, D SGD+MASSIVE+typo copies.
Each model is tuned on the combined SGD + MASSIVE dev set, then scored once on SGD test, MASSIVE test and the
hand-written messy test set. The messy set is never used for training or tuning. The Linear SVM learning curve is
trained on SGD only, tuned on SGD dev. Set D's models are saved to artifacts/classic/ for extract.py. Writes the
summary to results/classic_results.json and the per-row messy predictions to reports/classic_messy_predictions.csv.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from slots import (DictionaryTagger, add_typos, describe, featurize, fit_vectorizer, predict, read_jsonl, slot_f1,
                   span_scores)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
SEED = 42
TYPO_RATE = 0.15
SET_D_LABEL = "D: SGD+MASSIVE+typos"
GRIDS = {
    "MultinomialNB": ("alpha", [0.1, 0.3, 1.0], lambda v: MultinomialNB(alpha=v)),
    "LogisticRegression": ("C", [1, 3, 10, 30], lambda v: LogisticRegression(C=v, max_iter=2000, random_state=SEED)),
    "LinearSVC": ("C", [0.1, 0.3, 1, 3], lambda v: LinearSVC(C=v, random_state=SEED)),
}
SAVE_NAMES = {"MultinomialNB": "naive_bayes", "LogisticRegression": "logistic_regression", "LinearSVC": "linear_svm"}
FRACTIONS = [0.1, 0.25, 0.5, 1.0]
SLOTS = ["city", "date", "event_name"]


def load(name):
    return read_jsonl(ROOT / "data" / "clean" / f"{name}.jsonl")


def typo_copy(rows):
    """Copy each row with typo noise on its tokens; a fresh seeded rng keeps the copy reproducible."""
    rng = random.Random(SEED)
    return [{**r, "id": r["id"] + ":typo", "tokens": add_typos(r["tokens"], rng, TYPO_RATE)} for r in rows]


def set_d_rows():
    """Training set D: SGD, MASSIVE and a typo copy of each. train_bert.py trains on the same rows."""
    sgd, massive = load("train"), load("massive_train")
    return sgd + massive + typo_copy(sgd) + typo_copy(massive)


def training_sets():
    """The four training sets {label: rows}, labelled A to D as in the output."""
    sgd, massive = load("train"), load("massive_train")
    return {
        "A: SGD": sgd,
        "B: SGD+MASSIVE": sgd + massive,
        "C: SGD+typos": sgd + typo_copy(sgd),
        SET_D_LABEL: set_d_rows(),
    }


def load_evaluation():
    """The dev rows (SGD + MASSIVE) and the test sets {name: rows}. The messy set is only ever scored."""
    dev_rows = load("dev") + load("massive_dev")
    tests = {"SGD test": load("test"), "MASSIVE test": load("massive_test"),
             "messy": read_jsonl(ROOT / "data" / "messy" / "messy_test.jsonl")}
    return dev_rows, tests


def fit(model, feats, rows):
    vec, X = fit_vectorizer(feats)
    model.fit(X, [tag for r in rows for tag in r["tags"]])
    return vec, model


def tune(label, model_name, rows, feats, dev_rows, dev_feats):
    """Fit every grid value of model_name; return the best by dev span F1 (the first one wins ties)."""
    param, values, make = GRIDS[model_name]
    best = None
    for value in values:
        vec, model = fit(make(value), feats, rows)
        dev = span_scores([r["tags"] for r in dev_rows], predict(vec, model, dev_feats))
        print(f"{label} {model_name} {param}={value}: dev F1 {dev['f1']:.4f}", flush=True)
        if best is None or dev["f1"] > best["dev"]["f1"]:
            best = {"setting": f"{param}={value}", "dev": dev, "vec": vec, "model": model}
    return best


def score_tests(tests, predict_tags):
    """Span scores for every test set, and the messy predictions. predict_tags(name) gives the tags for one set."""
    scores, messy_pred = {}, None
    for name, rows in tests.items():
        pred = predict_tags(name)
        scores[name] = span_scores([r["tags"] for r in rows], pred)
        if name == "messy":
            messy_pred = pred
    return scores, messy_pred


def prediction_rows(set_label, model_name, messy_rows, pred):
    return [{"training_set": set_label, "model": model_name, "id": r["id"], "text": r["text"],
             "gold_spans": describe(r["tokens"], r["tags"]), "predicted_spans": describe(r["tokens"], p)}
            for r, p in zip(messy_rows, pred)]


def run_training_set(label, rows, dev, tests, test_feats):
    """Tune and score the three classic models and the dictionary baseline trained on one set.

    dev is (dev rows, their features). Return (result entries, messy prediction rows, tuned winners by model name,
    dictionary tagger). Only set D's winners and tagger are saved.
    """
    dev_rows, dev_feats = dev
    messy_rows = tests["messy"]
    feats = featurize(rows)
    results, predictions, winners = [], [], {}
    for model_name in GRIDS:
        best = tune(label, model_name, rows, feats, dev_rows, dev_feats)
        scores, messy_pred = score_tests(tests, lambda name: predict(best["vec"], best["model"], test_feats[name]))
        results.append({"training_set": label, "model": model_name, "setting": best["setting"],
                        "dev_f1": best["dev"]["f1"], "scores": scores})
        predictions += prediction_rows(label, model_name, messy_rows, messy_pred)
        winners[model_name] = best

    tagger = DictionaryTagger(rows)
    scores, messy_pred = score_tests(tests, lambda name: [tagger.tag(r["tokens"]) for r in tests[name]])
    dev_pred = [tagger.tag(r["tokens"]) for r in dev_rows]
    results.append({"training_set": label, "model": "dictionary baseline", "setting": "longest",
                    "dev_f1": span_scores([r["tags"] for r in dev_rows], dev_pred)["f1"], "scores": scores})
    predictions += prediction_rows(label, "dictionary baseline", messy_rows, messy_pred)
    return results, predictions, winners, tagger


def save_models(save_dir, winners, dictionary):
    """Write each set-D model as {model, setting, vectorizer, estimator} and the dictionary baseline as {tagger}."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for model_name, best in winners.items():
        joblib.dump({"model": model_name, "setting": best["setting"], "vectorizer": best["vec"],
                     "estimator": best["model"]}, save_dir / f"{SAVE_NAMES[model_name]}.joblib")
    joblib.dump({"model": "dictionary baseline", "setting": "longest", "tagger": dictionary},
                save_dir / "dictionary.joblib")


def run_learning_curve(sgd_rows, sgd_dev_rows, sgd_test_rows):
    """Linear SVM tuned on SGD dev, trained on growing random subsets of SGD train, scored on SGD test.

    Return (setting, points) where each point is {fraction, sentences, test_f1}.
    """
    feats = featurize(sgd_rows)
    best = tune("learning curve", "LinearSVC", sgd_rows, feats, sgd_dev_rows, featurize(sgd_dev_rows))
    test_feats = featurize(sgd_test_rows)
    order = random.Random(SEED).sample(range(len(sgd_rows)), len(sgd_rows))
    points = []
    for fraction in FRACTIONS:
        n = round(fraction * len(sgd_rows))
        idx = order[:n]
        vec, model = fit(clone(best["model"]), [feats[i] for i in idx], [sgd_rows[i] for i in idx])
        scores = span_scores([r["tags"] for r in sgd_test_rows], predict(vec, model, test_feats))
        points.append({"fraction": fraction, "sentences": n, "test_f1": scores["f1"]})
        print(f"learning curve {fraction:.0%} ({n} sentences): test F1 {scores['f1']:.4f}", flush=True)
    return best["setting"], points


def save_outputs(report, predictions):
    """Write the summary JSON to results/ and the per-row messy predictions CSV to reports/."""
    RESULTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "classic_results.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(REPORTS_DIR / "classic_messy_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)


def print_table(results):
    head = (f"{'training set':<22}{'model':<20}{'setting':>10}{'SGD F1':>9}{'MASSIVE F1':>12}{'messy F1':>10}  "
            + "  ".join(f"{s:<10}" for s in SLOTS))
    print(head)
    print("-" * len(head))
    for r in results:
        s = r["scores"]
        per_slot = "  ".join(f"{slot_f1(s['messy'], slot):<10}" for slot in SLOTS)
        print(f"{r['training_set']:<22}{r['model']:<20}{r['setting']:>10}{s['SGD test']['f1']:>9.3f}"
              f"{s['MASSIVE test']['f1']:>12.3f}{s['messy']['f1']:>10.3f}  {per_slot}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-dir", type=Path, default=ROOT / "artifacts" / "classic",
                        help="where the set-D models are saved")
    args = parser.parse_args()
    started = time.perf_counter()

    dev_rows, tests = load_evaluation()
    dev = (dev_rows, featurize(dev_rows))
    test_feats = {name: featurize(rows) for name, rows in tests.items()}

    results, predictions, set_d = [], [], None
    for label, rows in training_sets().items():
        set_results, set_predictions, winners, tagger = run_training_set(label, rows, dev, tests, test_feats)
        results += set_results
        predictions += set_predictions
        if label == SET_D_LABEL:
            set_d = (winners, tagger)
    save_models(args.save_dir, *set_d)
    print(f"saved set-D models to {args.save_dir}", flush=True)

    curve_setting, curve = run_learning_curve(load("train"), load("dev"), tests["SGD test"])
    seconds = time.perf_counter() - started
    report = {"seed": SEED, "typo_rate": TYPO_RATE, "dev": "SGD dev + MASSIVE dev", "results": results,
              "learning_curve": {"model": "LinearSVC", "setting": curve_setting, "dev": "SGD dev",
                                 "test": "SGD test", "points": curve},
              "run_seconds": seconds}
    save_outputs(report, predictions)

    print()
    print_table(results)
    print("\nlearning curve (Linear SVM, SGD only, test F1):")
    for point in curve:
        print(f"  {point['fraction']:>5.0%}  {point['sentences']:>5} sentences  {point['test_f1']:.4f}")
    print(f"\nrun time: {seconds:.1f}s")


if __name__ == "__main__":
    main()
