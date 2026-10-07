# AGENTS.md

Read [README.md](README.md) first. [PLAN.md](PLAN.md) records the completed rewrite.

## Project and course context

Solo COE025 Natural Language Processing Project 1: Text Classification (Fall
25–26, Asst. Prof. Yiğit Bekir Kaya). Core: classify human-localized MASSIVE tr-TR
requests into 60 intents. Slots are a removable token-classification extension.
No event search, request execution, frontend framework, or product integration.
Bi’ Plan / event_recommender is out of scope: import no code, data or settings.

Course: at least Naive Bayes and one other taught method, a public licensed
1000+ example dataset, benchmark comparison. GitHub repo due before the Week 4
presentation: 3 minutes plus 2 minutes Q&A. Solo by teacher arrangement; add no
team names. Rubric: technological depth 40%, originality 10%, presentation 50%.
Prefer simple, explainable TF-IDF, n-grams, NB, LR and SVM. No new dependencies,
services or neural models unless requested.

Status: NB/LR/SVM word and character models, majority, official splits, accuracy
and macro F1, per-intent metrics, scenario confusion, error analysis and paper
benchmark are met. MASSIVE v1.0 is public CC BY 4.0 with 16,521 requests.
The optional slot extension has NB/LR/SVM, BIO repair and exact span metrics.
Jev remains a core pretrained comparison method but is **not evaluated**.

## Commands and reproduction

Python 3.13; standard library, scikit-learn, matplotlib, joblib. Use the lock file.
Run every command from the repo root. Generated data/artifacts/reports and .env
are ignored; copy selected reports into evidence deliberately, never commit them
in their generated locations.

```powershell
python -m pip install -r requirements-lock.txt
python prepare_massive.py
python train.py
python train_slots.py                 # optional extension
python compare_methods.py            # no paid calls
python app.py                        # 127.0.0.1:8011 only
python app.py --no-slots
python -m unittest discover -s tests -v
```

Seed 42, official 11,514 train / 2,033 dev / 2,974 test. Selected character SVM:
dev accuracy/F1 83.42%/0.8112; test 82.95%/0.7898. Demo character LR test
79.15%/0.7534. Slot LR dev F1 0.6586, test 0.6370, joint exact match 52.86%.
40-request sample: Parser 7/40, LR 33/40, Jev not evaluated. Two fresh output
runs match; timestamps/request milliseconds are not deterministic.

## Layout

| Path | Role |
|---|---|
| prepare_massive.py | Verified pinned archive, only Turkish JSONL/license extraction, BIO alignment and audit; sklearn-free |
| classifier.py | 60 LABELS, normalization, seven candidate definitions, artifact validator, classification |
| train.py | Official split validation, dev weighting checks/selection, frozen test evaluation, reports |
| parser.py | Conservative train-derived seven-intent rules, coverage and answered accuracy |
| jev.py | Optional paid Jev Choice client, server secrets, durable attempt ledger and validation |
| method_comparison.py / compare_methods.py | Same original input across Parser, selected LR, optional Jev; fixed 40-case sample |
| slots.py / train_slots.py | Removable token models, BIO repair, exact spans, separate artifact/reports |
| app.py / demo.html | Loopback HTTP demo, dev examples, safe text and optional highlights |
| evidence/ | Deliberate report/audit/reproduction snapshots for graders |
| RESULTS.md / PRESENTATION.md | Evidence-aligned results and 3-minute solo talk |
| tests/ | unittest fixtures without prepared data, trained artifacts or live Jev |

## Invariants

- Keep the pinned MASSIVE v1.0 URL/SHA256. Verify before reading/extracting; extract
  only tr-TR.jsonl and LICENSE as regular files, with tarfile filter="data".
- classifier.LABELS is the fixed 60-intent source of truth, checked against data.
  OUTPUT_LABELS adds unclear, never a training class. Keep Jev criteria consistent.
- Preserve official partitions and every row, including duplicates. Require all
  intents in train and non-empty dev/test. Dev lacks audio_volume_other; test lacks
  cooking_query. Fixed 60-label macro F1 includes zero for absent classes. Do not
  reinstate performer groups, deduplication or a minimum per-class split size.
- Keep MIN_ROWS=1000 and the explicit small-prototype flag in train.py. Validate
  ids, text, labels, scenario and partition. Audit cross-partition/mixed-label texts.
- Intent selection uses dev macro F1; slot selection uses dev exact-span micro F1.
  Refit intent winners on train alone, never train+dev. Test is scored once per
  candidate after selection in each reproducibility run. Never tune on test.
- Write Parser intent/slot rules from train only. Keep Parser and learned outputs
  independent. Zero fitted TF-IDF features return unclear with unknown_terms.
- ARTIFACT_TASK=intent_classification, DATASET_DOMAIN=massive_tr. Validator and
  train artifact schemas must agree. model.joblib is the overall winner;
  lr_model.joblib is dev-selected LR for the demo/comparison. Joblib is trusted-only.
- Slot artifacts/reports stay separate. Core must work after deleting slots.py,
  train_slots.py and tests/test_slots.py. No mandatory core import of slots.
- BIO tags align 1:1 to whitespace tokens. Reject text mismatches/inside-word
  boundaries and audit them; preserve official suffixes/annotation inconsistencies.
  Repair I-x after O or another type as B-x. Score exact type and token boundaries.
- Quote paper reference numbers, not reproduced results. Paper full training uses
  all 51 locales and pretraining; ours uses Turkish only. Compare complete official
  Turkish test, never the 40-case sample as the paper benchmark. Report accuracy
  next to macro F1. No hand-reviewed or production-data claim for MASSIVE.

## Jev, secrets, code and tests

Jev must remain. No live calls during planned implementation; later calls need
explicit user authorization. JEV_MAX_CALLS defaults to 0, range 0–50. Each attempt
exclusively reserves reports/jev_calls/NNN.json. Failures count, no retries, and
never delete receipts to reset a budget. Keep the 8,000-byte cap and pinned model.
Tests use injected transport and temporary ledgers, never live network or the
real receipts directory. Partial provider runs have null accuracy.

The key stays server-side in ignored .env; do not return, log, or store it in
receipts. app.py never logs request bodies and binds only to 127.0.0.1. Demo builds
text with textContent/createTextNode and highlights with createElement spans;
never use innerHTML. Python offsets are code points; JS highlighting must use
Array.from(text), so emoji do not shift boundaries.

Match plain modules and short functions; type hints in classifier/train. Raise
readable ValueError, surfaced as CLI errors or HTTP 400. Turkish-aware case
normalization always maps I→ı / İ→i; Parser regexes use diacritic-stripped ASCII.
Use LF/UTF-8. Add dependencies only when asked; pin both requirements files.
Update README, RESULTS, PRESENTATION and data/README when behavior/numbers change.

Use unittest, not pytest. CI runs the lock file and suite on Ubuntu/Python 3.13.
Test data validation, official splits, dev-only selection, artifact validation,
Parser changes, Jev budgets and response validation. Slot tests are removable.

## Open author items

- Collect and label 30–50 real-person requests as a separate transfer test.
- Decide on an approved paid Jev run of the fixed 40 comparison requests.
- Ask the teacher whether the slot extension fits Project 1. If rejected, remove
  its files/tests and slot slide; core classification remains intact.
- Update presentation.pptx manually from PRESENTATION.md; this rewrite did not
  edit it. Add author name and arrange teacher GitHub access before presentation.
- Local CI-equivalent tests passed; no remote CI run was triggered here.
