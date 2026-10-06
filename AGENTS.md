# AGENTS.md

Guidance for coding agents working in this repository. Read [README.md](README.md) first for the user-facing overview.

## Project

A COE025 course project that classifies public event listing titles as `concert`, `theatre`, or `stand_up`. The data is the Gametime ticket listings dataset (CC BY-NC 4.0) on Hugging Face and Kaggle. The teacher requires a public dataset. Three methods run on the same input: rule-based Parser, trained Logistic Regression, and optional Jev (paid TypeSafe API). Any method may abstain with `unclear`, which is not a training class. It classifies categories only. It does not search events or extract dates, locations, or budgets.

Plain Python 3.13 scripts with no package layout, no web framework, and no frontend build. Dependencies are scikit-learn, matplotlib, and joblib.

## Course context

This repository is **Project 1: Text Classification** for COE025 Natural Language Processing (Fall 25-26, Asst. Prof. Yiğit Bekir Kaya). The requirements come from the Week 1 and Week 2 slides:

- Teams of 5 by default. **This project is solo by the teacher's arrangement.** Do not add team names. Choose one classification problem.
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
- Met: public dataset with 3,506 labeled titles and a checked license. Keep the 1,000-row check in `train.py`.
- **Weak: benchmark comparison.** Reference points are the majority baseline and the 36-title same-input comparison from the test split. A complete, approved Jev run on those 36 titles would give the pretrained reference point.

**Jev is a core part of the project. Do not remove it or suggest removing it.** The central story is that three approaches classify the same input: hand-written rules (Parser), a classical model trained on labeled data (Logistic Regression), and a pretrained model (Jev). The teacher likes Jev. Keep `jev.py` simple and well tested rather than cutting it.

**Stay in Project 1 scope, and keep the code simple.** The teacher grades the code and a 3-minute talk. Prefer methods taught in class (Naive Bayes, Logistic Regression, SVM, TF-IDF, n-grams) and plain, explicit code over clever abstractions. Do not add features from event_recommender, such as dates, places, budgets, or search. Do not add neural models, new services, or new dependencies unless the user asks. Before you add code, ask whether it helps a rubric item and whether a student can explain it in Q&A.

## Relation to event_recommender

This project is a standalone subset of the author's main product, **event_recommender** ("Bi' Plan", at `C:\Users\efeba\event_recommender`, a Node/TypeScript app for Istanbul events). The main product parses a whole request: dates, places, budgets, categories, negations, and age limits. It also uses Jev through TypeSafe. This repository isolates only the **event-category** part of that job: tagging event listings by category. An earlier version classified synthetic user requests; `data/request_challenge.json` remains from it as a transfer test.

- Do not import code, data, or settings from event_recommender. The README promises no dependency. Treat it only as a domain reference.
- Labels differ. The main product's `Category` type in `web/parser/contract.ts` is `concert | theatre | standup | workshop | exhibition`. Here the labels are `concert`, `theatre`, and `stand_up`, plus the `unclear` abstain output. Do not rename the labels here to match it unless the user asks.
- Follow the same cost discipline as the main product: no paid calls without approval, no automatic retries, and secrets stay in ignored local files.

## Commands

Run every command from the repository root. Scripts use relative default paths such as `data/events.csv`, `artifacts/`, and `reports/`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python prepare_events.py                    # downloads Gametime; data/events.csv + events.audit.json
python train.py                             # artifacts/model.joblib + reports/ (incl. comparison_set.json)
python compare_methods.py                   # reports/method_comparison.json
python compare_methods.py --data data/request_challenge.json --output reports/request_transfer.json
python app.py                               # http://127.0.0.1:8011
python -m unittest discover -s tests -v     # what CI runs
```

`prepare_events.py` reads a pinned Hugging Face commit (`REVISION`). Do not move it to `main`: the source keeps only 30 days and changes daily. `train.py` rejects fewer than 1,000 rows unless `--allow-small-prototype` is passed. Keep that check in place.

The pipeline is deterministic with seed 42. A clean run should reproduce `evidence/`: 3,506 rows, Logistic Regression (characters) test macro F1 0.5765, Parser 7/36, Logistic Regression 23/36, and on requests Parser 24/40, Logistic Regression 12/40.

## Layout

| Path | Role |
|---|---|
| `prepare_events.py` | Downloads Gametime events at a pinned revision, maps categories, cleans titles, groups by performer. Does not import sklearn. |
| `classifier.py` | `LABELS`, `normalize_text`, candidate pipelines, artifact validation, `classify_request` |
| `train.py` | CSV validation, group-disjoint 60/20/20 split, validation-based selection, reports, 36-title comparison set |
| `parser.py` | Regex rules with negation and exclusion handling. Independent of the learned model. |
| `jev.py` | Optional Jev client, call budget ledger, and response validation |
| `method_comparison.py` | Runs all three methods on one input (`compare_request`) |
| `compare_methods.py` | Scores the three methods on `reports/comparison_set.json`, or another set with `--data` |
| `app.py`, `demo.html` | Single-threaded `http.server` demo with vanilla JS. Binds to 127.0.0.1 only. |
| `evidence/` | Checked-in snapshots of reports for graders: metrics, confusion matrix, test errors, and method comparison |
| `RESULTS.md`, `PRESENTATION.md` | Numbers and talk outline. They must match `evidence/`. |
| `data/request_challenge.json` | 40 inspected AI-authored requests, used as a transfer test. Not a blind test set. |

Git ignores `data/*.csv`, `data/*.json` (except the challenge file), `artifacts/`, `reports/`, and `.env`. Do not commit generated outputs. To update evidence, copy the regenerated files from `reports/` into `evidence/` deliberately.

## Invariants

Preserve these unless the user asks to change them.

- **Labels.** `classifier.LABELS` is the source of truth for training classes; `OUTPUT_LABELS` adds `unclear`. `prepare_events.py` keeps its own copy so it can run without sklearn, so update both together. The labels also appear in `jev.CRITERIA` and `demo.html`.
- **No leakage.** Titles by the same first performer share a `group_id`, and `split_data` keeps groups out of more than one partition. Do not tune parser rules on test or comparison titles. Validation macro F1 selects the model, and test data never does. `read_csv` rejects mixed-label groups and normalized duplicates.
- **Methods stay independent.** Parser rules must not override or post-process Logistic Regression, and the reverse also applies. The only learned-side guard returns `unclear` when the input has no fitted features (`reason: unknown_terms`).
- **Artifact schema.** `classifier._validated_model` checks the keys, `task`, `dataset_domain`, and labels. If you add a field to the artifact in `train.py`, keep the validator consistent. `method_comparison` requires a `logistic_regression_*` model.
- **Joblib is trusted-only.** Load only artifacts made locally by `train.py`.
- **Honest reporting.** Labels are Gametime's categories, not hand-reviewed. Report macro F1 next to accuracy, since 73% of titles are concerts. Never call the 40 request cases a blind benchmark. Jev is "not evaluated" unless a complete live run exists. Partial runs get `accuracy: null` on purpose.

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

- `presentation.pptx` still shows the synthetic-request version. Update it to match `PRESENTATION.md`.
- Add a benchmark comparison that the teacher accepts. A complete Jev run on the 36 titles is the planned one.
- The author's name and teacher access to the GitHub repo are still pending. The repo is due before the Week 4 presentation.
