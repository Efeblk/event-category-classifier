# Plan: move to MASSIVE Turkish (intent core, slot extension)

Handoff plan for the implementing agent. Read `AGENTS.md` first. This plan replaces the event-title task. Where it conflicts with `AGENTS.md`, this plan wins, and step 8 updates `AGENTS.md` to match.

## Why

The author wants real human inputs. The Gametime titles are written by websites, and `data/request_challenge.json` was written by an AI. MASSIVE has real crowd-written requests and a published per-locale benchmark, which fixes the weak "compare with benchmarks" item.

Priority order: **COE025 Project 1 requirements first.** Bi' Plan (event_recommender) is only a side benefit. Do not import from it and do not shape the core task around it.

- **Core (graded):** intent classification, which is plain text classification.
- **Extension:** slot filling as token classification. It adds technical depth, and Bi' Plan needs it. If the teacher rejects the extension, it must be removable without touching the core.

## Dataset facts (verified 2026-10-07)

- Archive: `https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz`
  - sha256 `7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8` (about 39.5 MB)
  - Use v1.0. The leaderboard and paper use it, and v1.1 only adds Catalan.
- We need only `1.0/data/tr-TR.jsonl` and `1.0/LICENSE`. The license is CC BY 4.0, copyright Amazon.com Inc.
- tr-TR rows: train 11,514, dev 2,033, test 2,974. Each row's `partition` field says which split it belongs to.
- Fields:
  - `id`, `locale`, `partition`
  - `scenario`, `intent` (60 values, written as `scenario_intent`)
  - `utt`: the raw text
  - `annot_utt`: the text with slots written as `[slot_type : words]`
  - `worker_id`, `slot_method`, `judgments`
- There are 55 slot types. The top tr-TR training counts are `date` 1797, `place_name` 1053, `event_name` 996, `person` 861, `time` 792, `media_type` 478 and `timeofday` 234.
- Turkish quirks:
  - Slot spans usually include suffixes ("toplantıyla", "pawel'i").
  - Some requests start with the wake word "olly".
  - Some intents are tiny in train (`cooking_query` 4, `music_dislikeness` 14, `audio_volume_other` 18).
- Benchmark: FitzGerald et al. 2022, arXiv:2204.08582v2, Appendix D, Tables 7–9. tr-TR test results with full training data:

  | Model | Intent acc | Slot F1 | Exact match |
  |---|---:|---:|---:|
  | XLM-R base | 86.3 ±1.2 | 74.9 ±0.7 | 65.2 ±1.7 |
  | mT5 encoder-only | 87.1 ±1.2 | 76.1 ±0.7 | 67.7 ±1.7 |
  | mT5 text-to-text | 86.1 ±1.2 | 77.9 ±0.6 | 68.1 ±1.7 |

  Compare only on the full official tr-TR test split with all 60 intents. Otherwise the numbers are not comparable. Report these as quoted reference numbers, not reproduced ones.

A local copy for inspection is in `C:\Users\efeba\massive_dl` (outside the repo). Delete it when `prepare_massive.py` works. Treat the downloaded data as untrusted: extract with the tarfile `filter="data"` and read only the two members we need.

## Constraints that stay

- Keep: standard library plus scikit-learn, matplotlib and joblib, with no new dependencies; seed 42; `unittest`; LF line endings.
- Keep `normalize_text` Turkish-aware. Parser patterns stay ASCII on diacritic-stripped text.
- Keep the 1,000-row minimum check (`MIN_ROWS`) in `train.py`.
- Keep Jev:
  - The ledger, `JEV_MAX_CALLS` (default 0, maximum 50), no retries, and server-side secrets all stay.
  - **Make no real Jev calls.** Tests use an injected transport and a temporary ledger.
- Keep `unclear` as the abstain output. It is not a training class.
- Selection uses dev only. Test is scored once per model after selection, and nothing is tuned on test. Parser rules are written from train examples only.
- Commit no generated outputs. Copy files into `evidence/` deliberately.
- Keep the code plain enough that a student can explain every piece in a 2-minute Q&A.

## Steps

Work in this order and commit after each step. Each step must leave `python -m unittest discover -s tests -v` passing.

### 1. Data: `prepare_massive.py` (replaces `prepare_events.py`)
- **Download and check:**
  - Download the archive with `urllib` and check the pinned sha256.
  - Extract only `tr-TR.jsonl` and `LICENSE`.
  - Cache the archive under `data/` (ignored), and skip the download when the cached hash matches.
- **Write `data/massive_tr.jsonl`,** one row per utterance:
  - The fields `id`, `partition`, `intent`, `scenario` and `text` (`utt`).
  - `tokens`: whitespace tokens of the utterance.
  - `tags`: BIO tags, parsed from `annot_utt` and aligned 1:1 with the tokens.
  - Fail loudly if the de-annotated `annot_utt` does not match `utt`, or if a slot boundary falls inside a word. If boundaries inside words do occur, count them in the audit and document the rule you chose instead of guessing silently.
- **Write `data/massive_tr.audit.json`** with:
  - counts per partition, intent and slot type
  - the license, the source URL and the sha256
  - the number of normalized-text duplicates that appear in more than one partition, reported but not dropped, so the test set stays the official one
- **Keep it sklearn-free,** as `prepare_events.py` was.
- **Update `.gitignore`** if it needs new patterns.
- **Delete** `prepare_events.py`, `tests/test_events.py` and `data/request_challenge.json`. The challenge file's labels belong to the old task. The author will collect a new set from real people later (see "Later").
- **Update `data/README.md` and `data/LICENSE`** for MASSIVE.

### 2. Intent core: `classifier.py`, `train.py`
- **Labels:** `LABELS` becomes the 60 intents, read from the data and checked against a constant tuple in `classifier.py`, so the artifact validator can verify them. Set `ARTIFACT_TASK = "intent_classification"` and `DATASET_DOMAIN = "massive_tr"`.
- **Candidates:** keep `make_models` as it is (dummy, NB, LR and LinearSVC, each with word 1–2 and char_wb 3–5 features). Check that `class_weight="balanced"` still makes sense with 60 classes. Keep it unless dev macro F1 says otherwise, and report the decision.
- **Split:**
  - Use the official partitions. Remove the group-disjoint 60/20/20 split and its group checks.
  - Keep the duplicate and label sanity checks that still apply.
  - Remove `MIN_CLASS_ROWS_PER_SPLIT` if the official split violates it (`cooking_query`), or relax it and say why.
- **Selection and test:** select by dev macro F1, refit the winner on train only (not train+dev, to match the paper's protocol; document this), and score test once.
- **Reports in `reports/`:**
  - `metrics.json`: accuracy and macro F1 for every candidate on dev and test, the paper's numbers as `benchmark`, and a per-intent report for the winner.
  - Confusion: a 60×60 matrix is unreadable. Plot an 18×18 **scenario-level** confusion matrix, and list the top 15 confused intent pairs in the JSON.
  - `errors.csv`: test errors with text, gold and predicted.
  - `comparison_set.json`: 40 test requests sampled with seed 42, with Jev's budget in mind (40 of the 50 allowed calls, leaving slack).
- **Artifact:** `artifacts/model.joblib` keeps a validated schema. Update `_validated_model` to match.
- **Tests:** rewrite `tests/test_train.py` and `tests/test_models.py` with small fixtures for official-split handling, selection on dev only, and artifact validation.

### 3. Parser on intents: `parser.py`
- **Rules:** write keyword rules, from train examples only, for a handful of clear intents (for example `alarm_set`, `weather_query`, `play_music`, `recommendation_events`, `calendar_set`, `datetime_query`, `iot_hue_lightoff`). Everything else returns `unclear`.
- **Old rules:** drop the old concert/theatre/stand-up patterns and the negation logic unless they still serve an intent.
- **Scoring:** report coverage (share not `unclear`), accuracy on the requests it answered, and overall accuracy, so the Parser is judged fairly as a high-precision baseline.
- **Tests:** rewrite `tests/test_parser.py`.

### 4. Jev on intents: `jev.py`, `method_comparison.py`, `compare_methods.py`
- **Criteria:** `CRITERIA` becomes one short line per intent plus `unclear`. Check that the request body stays under the existing 8,000-byte cap. If it doesn't, shorten the descriptions; do not quietly raise the cap.
- **Versioning:** bump `PROMPT_VERSION`.
- **Live API:** read the TypeSafe docs (the `typesafe-ai` skill) to confirm the `choice` question shape still fits 61 options. Do not call the live API.
- **Comparison:** `compare_methods.py` scores Parser, the selected Logistic Regression model and (only with `--include-jev`) Jev on the 40 comparison requests. Jev is reported as "not evaluated" without a complete run.
- **Tests:** update `tests/test_method_comparison.py` (mock transport, temporary ledger).

### 5. Slot extension: `slots.py`, `train_slots.py`
- **Features:** one dict per token:
  - the lowercased word
  - its first 3–5 and last 2–4 characters (suffix features matter for Turkish)
  - flags for numbers, capitals and apostrophes
  - the previous and next 1–2 words, plus start and end of sentence
  - Vectorize with `DictVectorizer`.
- **Candidates:** a majority baseline (all `O`), `MultinomialNB` (or `BernoulliNB`, chosen on dev), `LogisticRegression` and `LinearSVC`. Each classifies one token at a time.
- **Decoding:** repair invalid BIO sequences (treat an `I-x` after `O` or after a different type as `B-x`). Keep the rule simple and document it.
- **Metrics:** written by hand, with no seqeval.
  - span-level micro precision, recall and F1, where a span counts only on an exact type and boundary match
  - per-slot F1
  - exact match: intent and all slots correct
    - Use the selected intent model for the intent part, so the number is comparable with the paper's.
- **Report and select:**
  - Highlight the slots that matter for events: `date`, `time`, `timeofday`, `place_name`, `event_name`, `artist_name`, `person`.
  - Select on dev slot F1 and score test once.
  - Error analysis: boundary errors (suffix spans) vs. type confusions (date vs. time).
- **Parser slots:**
  - Regex rules for `date`, `time` and `timeofday` (bugün, yarın, weekdays, "akşam dokuzda", "saat 5'te", …), written from train.
  - Score them only on those three slot types, and label that clearly.
- **Jev slots:** optional, and only if TypeSafe supports a suitable extraction question type. Otherwise leave it out and say so. Do not build a free-text parsing layer.
- **Isolation:**
  - Use a separate artifact (`artifacts/slot_model.joblib`) and separate reports (`reports/slot_metrics.json`, `reports/slot_errors.csv`).
  - Deleting `slots.py`, `train_slots.py` and their tests must leave the core working.
- **Tests:** `tests/test_slots.py` covers BIO parsing from `annot_utt`, BIO repair, span F1 on hand-built cases, and feature extraction.

### 6. Demo: `app.py`, `demo.html`
- **Behavior:** the user types a request. The demo shows the intent from Parser, Logistic Regression and (opt-in) Jev, and below that the extracted slots, with the extracted words highlighted per method.
- **Rules that stay:**
  - Bind to 127.0.0.1 only.
  - Do not log request bodies.
  - Build the DOM with `textContent` only; the highlighting uses `<span>` elements created with `createElement`.
- **Examples:** replace the example buttons with real tr-TR requests taken from **dev**, not test.

### 7. Run and evidence
- **Run the pipeline from a clean state:** `prepare_massive.py`, `train.py`, `train_slots.py`, `compare_methods.py`.
- **Evidence:** copy the reports into `evidence/` and delete the obsolete evidence files (`request_transfer.json` and the old ones). Check that a second clean run reproduces the same numbers.

### 8. Docs
- **Rewrite** `README.md`, `RESULTS.md`, `PRESENTATION.md`, `data/README.md` and `AGENTS.md` to match the new task and the new numbers. Keep them short and plain.
- **`RESULTS.md` must have:**
  - an honest benchmark table: our classical models vs. the paper's pretrained models
  - a note that we expect to score below them, and why: no pretraining, sparse features
  - Parser coverage and accuracy
  - "Jev: not evaluated" unless a complete live run exists
- **Talk outline in `PRESENTATION.md`:** 3 minutes, with intent as the core and one slide for slots.
- **AGENTS.md:** update "Course context → Status", "Layout", "Invariants", the reproduction numbers in "Commands", and "Open items".
- **Do not touch** `presentation.pptx`. The author updates it.

## Later (not for this agent)

- The author collects 30–50 real requests about Istanbul events from people and labels them as a transfer test. They replace the old AI-written challenge file.
- The author decides on a paid Jev run of the 40 comparison requests.
- The teacher confirms that the slot extension fits the scope. If the answer is no, delete step 5's files and the slot slide.

## Done when

- Tests pass locally and in CI.
- A clean run reproduces `evidence/`.
- `RESULTS.md` compares our numbers with the paper's tr-TR numbers on the official test split.
- No generated files or secrets are committed.
