# Turkish request intent classifier

Solo COE025 NLP Project 1. The graded core classifies a Turkish request into one
of **60 MASSIVE intents**, such as `alarm_set`, `weather_query` or `play_music`.
A removable extension labels whitespace tokens with **BIO slot tags**.
It classifies and annotates requests; it does not execute them or search events.

MASSIVE v1.0 is public, under **CC BY 4.0** (Amazon.com Inc.). Its Turkish requests
were localized by humans from English crowd-written SLURP utterances. These are
human-created benchmark requests, not Turkish production logs or AI-written cases.
See [the paper](https://arxiv.org/html/2204.08582v2) and [dataset details](data/README.md).

## Run

Python 3.13, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python prepare_massive.py
python train.py
python train_slots.py
python compare_methods.py
python app.py
```

On Linux/macOS, activate with `source .venv/bin/activate`. Preparation downloads
about 39.5 MB, verifies the pinned SHA256, and extracts only tr-TR and LICENSE with
`tarfile`'s data filter. A verified archive cache skips the download.
Data, model artifacts, reports and secrets remain ignored. Checked-in `evidence/`
is a deliberate snapshot for grading, not an input to training.

Open [localhost:8011](http://127.0.0.1:8011). Example buttons use official **dev**
requests. Parser, dev-selected Logistic Regression, and optional Jev receive the
same text. `unclear` is an abstention, never a training class. Slots are shown
separately with their actual method name and highlighted source words.

## Evaluation

Official partitions: **11,514 train / 2,033 dev / 2,974 test**. No resplitting or
removal of duplicate texts. Train contains all 60 intents; dev lacks
`audio_volume_other`, and test lacks `cooking_query`. Fixed 60-label macro F1
assigns zero to absent classes. Dev macro F1 selects the intent winner; dev exact
span F1 selects the slot winner. Every model fits on train alone, never train+dev.
Test is evaluated after selection; no rules or settings are tuned on test.

Intent candidates: majority, Naive Bayes, Logistic Regression and Linear SVM
with word 1–2 and character 3–5 TF-IDF features. Four additional LR/SVM variants
check balanced versus unbalanced weights on dev. The winner is character SVM:
**test accuracy 82.95%, macro F1 0.7898**. The demo uses character LR from a separate
`artifacts/lr_model.joblib`; the overall winner is `artifacts/model.joblib`.

The paper's Turkish intent reference is **86.3% for XLM-R**, with larger pretrained
models trained on all 51 locales. Our Turkish-only sparse models are not a
reproduction of that training setup. [Results and limitations](RESULTS.md) include
all candidates, Parser coverage, slot results, and the full benchmark table.

## Optional components

Slots: omit `train_slots.py`, use `python app.py --no-slots`, or remove `slots.py`,
`train_slots.py` and `tests/test_slots.py`. Intent training and comparison still
work. Slot artifacts/reports are separate. Jev slot extraction is not implemented:
TypeSafe exposes Choice, Noul and Score, not arbitrary span extraction.

Jev: **not evaluated**. No live calls were made during this rewrite. A future paid
run requires explicit author approval. Copy `.env.example` to `.env`, set the
server-only key and `JEV_MAX_CALLS` (default 0, maximum 50), then restart. The demo's
opt-in or `compare_methods.py --include-jev` makes one attempt per request, with
no retries. The 40-request comparison leaves ten attempts of slack. Failed
attempts count in the shared `reports/jev_calls/` ledger; never delete receipts.
Partial runs have no accuracy score. [TypeSafe Choice](https://docs.typesafe.ai/primitives/choice)
allows 61 options within the unchanged 8,000-byte request cap.

## Tests and author tasks

```powershell
python -m unittest discover -s tests -v
```

Tests use small fixtures, injected Jev transports and temporary ledgers. CI runs
this command on Ubuntu/Python 3.13. Two fresh output runs reproduced the evidence
metrics and predictions; see [reproduction checks](evidence/reproduction_check.json).

The author still needs to collect 30–50 real requests from people, decide on the
paid Jev comparison, ask the teacher about the slot extension, and update
`presentation.pptx` using [the 3-minute outline](PRESENTATION.md). The slide file is
unchanged. Add the author's name and give the teacher GitHub access before presenting.
