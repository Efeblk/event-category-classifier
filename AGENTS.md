# AGENTS.md

Read [README.md](README.md) first.

## Current scope

Solo COE025 NLP Project 1: **Spot the Manipulation**. Classify a provided English
news excerpt, with optional surrounding context, into the 14 SemEval-2020 Task 11
persuasion techniques. The user selected this replacement for the former Turkish
intent project. MASSIVE, token slots, event search and unrelated applications are
out of scope. Do not restore their files, numbers or requirements.

Use plain Python modules, standard library, scikit-learn, matplotlib and joblib;
no new dependencies, neural training or services without a user request. Use
unittest, UTF-8 and LF. Keep functions explainable and readable ValueError messages.

## Latest teacher rubric

Overall: presentation 50%, implementation 50%. Presentation: technical explanation
50%, fluency 25%, pace 25%; questions may adjust technical depth. Rehearse to
1:50, without reading. Implementation: correctness 20%, data novelty 30%, task
novelty 30%, rigor 20%. Publish a concise public GitHub repo and submit its URL
with the PPTX. Include raw and cleaned data, their license and source attribution,
preparation/training scripts, detailed README and a truthful contribution file.
Author name, student ID and the personal contribution record are deferred by the
user. The user requested the remaining teacher deliverables, including publication
of the current project to the existing public repository.
The user subsequently excluded presentation creation. Do not create or export
a PPTX unless the user requests it again; the existing outline is reference only.
Do not invent names, contributions or a guaranteed grade.

## Reproduction

Run commands from the repository root, Python 3.13:

```powershell
python -m pip install -r requirements-lock.txt
python prepare_data.py
python train.py
python compare_methods.py
python reproduce.py
python app.py
python -m unittest discover -s tests -v
```

Keep generated model artifacts, reports, private builds and secrets ignored.
Raw selected article/annotation files and cleaned examples are deliberately public
under data/. Copy only final report snapshots into evidence/.

## Data and evaluation invariants

- Pin and verify the original Zenodo archive SHA256 before reading or extracting.
  Extract whitelisted regular files only with tarfile filter="data". Keep original
  UTF-8 article bytes; offsets count Unicode code points, not bytes.
- classifier.LABELS is the fixed 14-technique source of truth. `unclear` is an
  inference abstention, never a training class. English lowercasing applies here.
- Use only the original archive's labeled training articles. Official dev has
  spans but no technique labels; this archive has no official labeled test data.
  Our article-grouped train/dev/test split is a new, explicitly documented protocol.
  Published official scores are contextual references, not directly comparable.
- Preserve every different technique annotation, including repeated boundaries
  with different labels. Remove exact duplicate annotation records only with an
  audit. Validate text, labels, IDs, bounds and context before fitting.
- Seed 42. Keep articles and identical complete inputs in one partition. Fit all
  TF-IDF vocabularies and classifiers on train alone. Require all 14 labels in
  train; dev/test may lack a label. MIN_ROWS=1000 and an explicit small-prototype
  flag remain. The original nine-candidate test benchmark is frozen under
  evidence/baseline/. The current upgrade is development-only: read
  data/cleaned/development.jsonl, which contains train/dev and wider context
  (1,000 source characters on either side). Never score the upgrade on the old
  test after using its error analysis; a fresh final evaluation is pending.
  Select models by dev macro F1 and leave every current test metric null.
- Report accuracy, micro F1, fixed-14-label macro F1, per-class precision/recall,
  confusion and failures. Examples shown by the demo come from dev, not test.
- Artifacts declare task=persuasion_technique_classification and
  dataset_domain=ptc_semeval2020, schema version 2, labels and provenance.
  evaluation_scope/context_margin must be dev_only/1000 for the upgrade or
  historical_baseline/350 for the explicit original baseline profile. Validators and
  training schemas agree. Joblib is trusted-only. model.joblib is the overall
  dev winner; method_models.joblib contains dev-selected NB/LR/SVM artifacts.
- The upgrade adds one explainable LR candidate: separate excerpt words,
  context words, excerpt characters and 22 scaled text-shape counts. Fit every
  vectorizer and scaler on train only. Numeric features alone must not suppress
  the demo's unknown-vocabulary abstention. Keep the original three cleaned
  benchmark files unchanged; development files are separate.
- No claim of automatic whole-article detection, truth checking, causal feature
  explanations, hand-reviewed project data or original scraping. The corpus's
  creators already cleaned it; describe our measured transformations honestly.

## Jev, secrets and demo

Jev remains an optional pretrained comparison, **not evaluated**. No live calls
while implementing; later paid calls require explicit user authorization.
JEV_MAX_CALLS defaults to 0, range 0-50. Each attempted call exclusively reserves
reports/jev_calls/NNN.json. Failures count, no retries, never delete receipts to
reset a budget. Keep the pinned model, 3,000-character combined input limit and
8,000-byte payload cap. Preflight every sample before any paid attempt. Tests use
injected transport and temporary ledgers, never real receipts or provider calls.
Incomplete provider comparisons have null accuracy and F1.

Keys stay server-side in ignored .env, never in logs, responses or receipts.
app.py binds only 127.0.0.1 and never logs request bodies. Demo text uses
textContent/createTextNode, never innerHTML. For offsets use Array.from(text).

## Collaboration and verification

The user explicitly requested subagents and cross-development. Divide file
ownership, then have agents review another component. Run the full unittest
suite, actual preparation/training, reproduction comparison and browser checks.
Update README, RESULTS, PRESENTATION and data/README when behavior/numbers change.
Keep grade-related limitations visible and author tasks separate from measured work.
