# AGENTS.md

Guidance for coding agents working in this repository. Read [README.md](README.md) first for the user-facing overview.

## Project

A COE025 course prototype that classifies Turkish and English event requests as `concert`, `theatre`, `stand_up`, or `unclear`. Three methods run on the same input: rule-based Parser, trained Logistic Regression, and optional Jev (paid TypeSafe API). It classifies requests only. It does not search events or extract dates, locations, or budgets.

Plain Python 3.13 scripts with no package layout, no web framework, and no frontend build. Dependencies are scikit-learn, matplotlib, and joblib.

## Course context

This repository is **Project 1: Text Classification** for COE025 Natural Language Processing (Fall 25-26, Asst. Prof. Yiğit Bekir Kaya). The requirements come from the Week 1 and Week 2 slides:

- Team of 5 students. Choose one classification problem.
- Implement at least **Naive Bayes plus one other method from class** (Logistic Regression or SVM).
- **Compare the results with benchmarks.**
- Use a dataset with **at least 1,000 examples and clear labels**, and check its license. The slides suggest HuggingFace, Kaggle, and UCI.
- Week 3 is implementation. The presentation is in **Week 4: 3 minutes plus 2 minutes of Q&A**. The **GitHub repo is due before the presentation**.

Grading rubric. Use it to decide what work matters.

| Part | Weight | Criteria |
|---|---:|---|
| Technological depth | 40% | Code quality and correctness, number of methods compared (more is better), feature engineering, a proper train/test split and metrics, error analysis and insights |
| Idea originality | 10% | Problem or dataset choice, novel feature combinations, domain application |
| Presentation | 50% | Clear problem and approach, technical communication, slides, Q&A, staying within 3 minutes |

Status against the requirements:

- Met: Naive Bayes, Logistic Regression, and Linear SVM, each with word and character features, plus a majority baseline. Also met: a group-disjoint split, macro F1 and per-class metrics, a confusion matrix, and error analysis in `RESULTS.md`.
- **Not met: dataset size.** There are 364 synthetic rows. Do not hide this or weaken the 1,000-row check in `train.py`.
- **Weak: benchmark comparison.** The only reference points are the majority baseline and the inspected 40-case set. No external benchmark exists.

**Stay in Project 1 scope, and keep the code simple.** The teacher grades the code and a 3-minute talk. Prefer methods taught in class (Naive Bayes, Logistic Regression, SVM, TF-IDF, n-grams) and plain, explicit code over clever abstractions. Do not add features from event_recommender, such as dates, places, budgets, or search. Do not add neural models, new services, or new dependencies unless the user asks. Before you add code, ask whether it helps a rubric item and whether a student can explain it in Q&A.

## Relation to event_recommender

This project is a standalone subset of the team's main product, **event_recommender** ("Bi' Plan", at `C:\Users\efeba\event_recommender`, a Node/TypeScript app for Istanbul events). The main product parses a whole request: dates, places, budgets, categories, negations, and age limits. It also uses Jev through TypeSafe. This repository isolates only the **event-category** part of that job as a course-sized classification task.

- Do not import code, data, or settings from event_recommender. The README promises no dependency. Treat it only as a domain reference.
- Labels differ. The main product's `Category` type in `web/parser/contract.ts` is `concert | theatre | standup | workshop | exhibition`. Here the labels are `concert`, `theatre`, `stand_up`, and `unclear`. Do not rename the labels here to match it unless the user asks.
- Follow the same cost discipline as the main product: no paid calls without approval, no automatic retries, and secrets stay in ignored local files.

## Commands

Run every command from the repository root. Scripts use relative default paths such as `data/requests.csv`, `artifacts/`, and `reports/`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python prepare_requests.py                  # data/requests.csv + requests.audit.json
python train.py --allow-small-prototype     # artifacts/model.joblib + reports/
python compare_methods.py                   # reports/method_comparison.json
python app.py                               # http://127.0.0.1:8011
python -m unittest discover -s tests -v     # what CI runs
```

`train.py` fails without `--allow-small-prototype` because the dataset has 364 rows and the course minimum is 1,000. Keep that check in place.

The pipeline is deterministic with seed 42. A clean run should reproduce `evidence/metrics.json`: Logistic Regression test macro F1 0.8195, Parser 23/40, Logistic Regression 25/40.

## Layout

| Path | Role |
|---|---|
| `prepare_requests.py` | Hand-written Turkish/English paraphrase families, written to CSV. Does not import sklearn. |
| `classifier.py` | `LABELS`, `normalize_text`, candidate pipelines, artifact validation, `classify_request` |
| `train.py` | CSV validation, group-disjoint 60/20/20 split, validation-based selection, reports |
| `parser.py` | Regex rules with negation and exclusion handling. Independent of the learned model. |
| `jev.py` | Optional Jev client, call budget ledger, and response validation |
| `method_comparison.py` | Runs all three methods on one input (`compare_request`) |
| `compare_methods.py` | Scores the three methods on `data/request_challenge.json` |
| `app.py`, `demo.html` | Single-threaded `http.server` demo with vanilla JS. Binds to 127.0.0.1 only. |
| `evidence/` | Checked-in snapshots of reports for graders: metrics, confusion matrix, test errors, and method comparison |
| `RESULTS.md`, `PRESENTATION.md` | Numbers and talk outline. They must match `evidence/`. |
| `data/request_challenge.json` | 40 inspected regression cases. Not a blind test set. |

Git ignores `data/*.csv`, `data/*.json` (except the challenge file), `artifacts/`, `reports/`, and `.env`. Do not commit generated outputs. To update evidence, copy the regenerated files from `reports/` into `evidence/` deliberately.

## Invariants

Preserve these unless the user asks to change them.

- **Labels.** `classifier.LABELS` is the source of truth. `prepare_requests.py` keeps its own copy so it can run without sklearn, so update both together. The labels also appear in `jev.CRITERIA` and `demo.html`.
- **No leakage.** Paraphrases and translations share a `group_id`, and `split_data` keeps groups out of more than one partition. Validation macro F1 selects the model, and test data never does. `read_csv` rejects mixed-label groups and normalized duplicates.
- **Methods stay independent.** Parser rules must not override or post-process Logistic Regression, and the reverse also applies. The only learned-side guard returns `unclear` when the input has no fitted features (`reason: unknown_terms`).
- **Artifact schema.** `classifier._validated_model` checks the keys, `task`, `dataset_domain`, and labels. If you add a field to the artifact in `train.py`, keep the validator consistent. `method_comparison` requires a `logistic_regression_*` model.
- **Joblib is trusted-only.** Load only artifacts made locally by `train.py`.
- **Honest reporting.** Scores come from synthetic data. Never describe them as real-user accuracy, and never call the 40 challenge cases a blind benchmark. Jev is "not evaluated" unless a complete live run exists. Partial runs get `accuracy: null` on purpose.

## Jev and secrets

- Calls cost money. Never make a real Jev call (`compare_methods.py --include-jev`, or **Include Jev** in the demo) unless the user explicitly asks.
- `JEV_MAX_CALLS` defaults to 0, which disables Jev, and accepts values up to 50. Each attempt reserves a receipt in `reports/jev_calls/NNN.json` with exclusive create. Failed attempts count, and there are no retries. Never delete receipts to reset the budget, and never add retry logic.
- The API key stays on the server. Do not return it from any endpoint, log it, or write it into receipts. `.env` must stay uncommitted.
- `app.py` deliberately does not log request bodies. Keep it that way.
- Tests must use an injected `transport` or a mock and a temp `ledger_dir`. They must never hit the network or `reports/jev_calls/`.

## Code style

- Standard library and sklearn only. Do not add dependencies without asking. If you do add one, pin it in both `requirements.txt` and `requirements-lock.txt`.
- Match the surrounding style: short modules, type hints in `classifier.py` and `train.py`, terse docstrings, and `ValueError` with user-readable messages that become `SystemExit` or HTTP 400.
- Text normalization must stay Turkish-aware (`I`→`ı`, `İ`→`i`). The Parser matches against diacritic-stripped text, so write its patterns in ASCII (`muzik`, `canli`, `degil`).
- `demo.html` builds the DOM with `textContent`. Do not insert user or model text with `innerHTML`.
- Files use LF line endings (`.gitattributes`). Prose files are UTF-8.
- Docs are deliberately short and plain. When behavior or numbers change, update `README.md`, `RESULTS.md`, `PRESENTATION.md`, and `data/README.md` in the same change.

## Testing

- `tests/` uses `unittest`, not pytest. CI (`.github/workflows/tests.yml`) runs it on Ubuntu with Python 3.13 and the lock file.
- Tests do not need a prepared dataset or a trained model. They build small fixtures in temp directories.
- Add or extend a test in the matching `tests/test_*.py` for parser rule changes, split or selection logic, artifact validation, and Jev budget or validation behavior.

## Open items (not code bugs)

- The dataset needs 1,000 or more examples with clear labels and a checked license, plus an unseen, human-reviewed evaluation set. Ask the teacher whether synthetic data is acceptable.
- Add a benchmark comparison that the teacher accepts.
- Team names (5 students) and teacher access to the GitHub repo are still pending. The repo is due before the Week 4 presentation.
