# Spot the Manipulation

Solo COE025 Natural Language Processing Project 1. Given a **selected English
news excerpt** and optional surrounding context, classify its persuasion technique:
loaded language, name-calling, fear appeals, slogans, false dilemmas and nine other
categories. The graded core compares **Naive Bayes, Logistic Regression and
Linear SVM**, with a majority baseline.

This is technique classification of a provided fragment. It does not locate new
propaganda spans, check whether a claim is true, or establish a writer's intention.
The entertaining demo lets you guess the technique before revealing independent
model predictions. It is a course experiment, not a moderation or fact-checking tool.

## Run locally

Use Python 3.13 and run every command from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python prepare_data.py
python train.py
python compare_methods.py
python app.py
```

On Linux/macOS activate with `source .venv/bin/activate`. Open
[127.0.0.1:8011](http://127.0.0.1:8011). The server binds only to loopback.
No frontend build step, neural model training or paid API is required.

Preparation verifies the pinned original archive before extracting selected
regular files. Public raw and cleaned data are deliberately included under
`data/`; rerunning the script reconstructs the same examples and partitions.
Model artifacts, generated reports, the download cache and local credentials are
ignored. Load joblib artifacts only if you created or otherwise trust them.

## Data and preprocessing

The [original SemEval-2020 Task 11 dataset](https://zenodo.org/records/3952415)
is licensed under **CC BY 4.0**. The selected release has 371 original labeled
training articles and 6,129 annotation rows. Our preparation removes one identical
annotation record and reconstructs **6,128 examples**, preserving every different
technique annotation. Fourteen articles have no annotated fragments; they remain
in the raw corpus but produce no invented negative examples.

The pipeline checks article IDs, the fixed 14 technique names, Unicode character
boundaries and exact original substrings. It preserves raw article bytes and
excerpt text, creates a whitespace-normalized representation and bounded context,
and records duplicates, conflicting labels and all transformations. See
[data/README.md](data/README.md), [source attribution](data/SOURCE.md), the
[dataset license](data/LICENSE), and `data/cleaned/audit.json`.

The creators already curated and corrected this corpus. We do not claim to have
scraped it or performed their cleaning. Our documented preparation is not a
promise of a particular data-novelty mark.

## Training and evaluation

The archive lacks official development/test technique gold labels. We create a
**custom article-grouped split of the labeled official training corpus**. Seed 42
assigns approximately 65/15/20% of article groups to train/dev/test:

| Partition | Annotation examples | Articles with annotations |
|---|---:|---:|
| Train | 3,795 | 231 |
| Development | 1,048 | 54 |
| Test | 1,285 | 72 |

Articles and identical excerpt/context inputs cannot cross partitions. Short
phrases can recur in different articles with different contexts; this is audited
and limits claims about independence. This is not a held-out publisher evaluation.

The original nine-candidate benchmark is preserved under
`evidence/baseline/`. Its error analysis motivated a simple upgrade: LR combines
excerpt words, character patterns, wider context and 22 text-shape counts.
These include length, punctuation, capitalization and exact phrase repetition.
No dependency or neural model was added.

**The current pipeline is development-only.** `prepare_data.py` additionally
reconstructs `development.jsonl` from only the existing train/dev articles, with
up to 1,000 source characters on each side. Original benchmark files and
partitions remain unchanged. `python train.py` compares ten candidates, fitting
every vocabulary, scaler and classifier on **train alone**. Development macro
F1 selects the overall model and the best variant for each method.

The upgrade reaches **57.35% dev accuracy / 0.3957 macro F1**, compared with
**49.24% / 0.3324** for the original selected model on dev. These are development
results, not an increase in test accuracy. Because the old test errors informed
the upgrade, a fresh final evaluation is pending. Current test metrics stay null.
To reproduce the historical nine-candidate protocol separately, run
`python train.py --profile baseline`; its outputs go to `reports/baseline/` and
`artifacts/baseline/` by default.

Report accuracy, micro F1 and macro F1 over the fixed 14 labels, along with
per-class precision/recall, a confusion matrix and errors. Some fragments have
more than one annotated technique; those annotation rows remain in the same
partition. Our ordinary annotation-row scoring differs from the competition's
matching procedure for repeated spans. Published competition scores are
**contextual references**, not a direct comparison with our custom test.

[RESULTS.md](RESULTS.md) records measured results and limitations. Demo examples
are selected from development data without filtering by correct predictions.
`compare_methods.py` now uses a fixed 40-case **dev** sample solely for a same-input
comparison. The original test benchmark remains separately labeled in the demo.

## Repository map

| Path | Purpose |
|---|---|
| `prepare_data.py` | Verified download, safe extraction, span reconstruction, cleaning audit and grouped split |
| `classifier.py` | Fixed label definitions, English normalization, feature/model candidates, artifact validation and predictions |
| `train.py` | Train-only fitting and dev-only selection/reports; explicit historical baseline profile |
| `reproduce.py` | Fresh temporary preparation/training; exact cleaned/report/prediction and saved-artifact checks |
| `method_comparison.py` | Same excerpt/context passed independently to NB, LR, SVM and optional Jev |
| `compare_methods.py` | Reproducible comparison report; optional paid calls require explicit opt-in |
| `jev.py` | Retained optional pretrained comparator, server secrets, response validation and durable attempt budget |
| `app.py` / `demo.html` | Local HTTP API and accessible guess-before-reveal demo |
| `data/raw/` | Original article text and annotation files, retained byte-for-byte |
| `data/cleaned/` | Original benchmark plus separate wider-context train/dev examples and audits |
| `data/README.md`, `data/SOURCE.md`, `data/LICENSE` | Preparation decisions, provenance, attribution and dataset reuse terms |
| `evidence/` | Current dev snapshots and separately frozen `baseline/` test benchmark |
| `tests/` | unittest fixtures for preparation, selection, artifacts, API and disabled/injected Jev |
| `.github/workflows/tests.yml` | Ubuntu/Python 3.13 lock-file installation and unittest suite |
| `.gitattributes` | LF source/documentation checkout and byte-preserving raw dataset files |
| `requirements.txt`, `requirements-lock.txt` | Direct dependencies and fully pinned reproduction environment |
| `LICENSE` | MIT license for project scripts; the dataset retains its separate CC BY 4.0 terms |
| `RESULTS.md` / `PRESENTATION.md` | Measured findings and a 1:50 solo presentation outline |
| `AGENTS.md` | Current scope and engineering invariants |
| `contributions/` | Guide for the required truthful solo contribution record; personal file pending |

## Optional Jev comparison

Jev is retained as a pretrained comparison and **has not been evaluated**.
Default `JEV_MAX_CALLS=0` disables calls. Later paid evaluation requires explicit
author authorization. No paid calls are made by training, tests or the default demo.

Copy `.env.example` to ignored `.env`; keep the key server-side. The pinned model
is `jev-1.13.0`, and the budget ranges from 0 to 50 shared attempts. Each attempt
reserves `reports/jev_calls/NNN.json` exclusively. Failures count; there are no
retries, and deleting receipts must not be used to reset the budget. Inputs are
limited to 3,000 combined excerpt/context characters and an 8,000-byte payload.
Partial provider runs have null scores. See the
[TypeSafe Choice documentation](https://docs.typesafe.ai/primitives/choice).

## Checks and submission

```powershell
python -m unittest discover -s tests -v
python reproduce.py
```

Tests use small fixtures, local HTTP servers, temporary ledgers and injected
provider responses. They require neither prepared datasets nor trained artifacts.
The demo inserts user text through textContent/createTextNode and does not log
request bodies. Displayed feature weights describe model contributions rather
than a human explanation or a calibrated guarantee of correctness.

The current local verification passes **60 tests**. A fresh temporary rebuild
matches all five cleaned files, four reports, ten candidate dev prediction arrays
and saved-model metadata/dev predictions. These checks make no provider calls.

The latest rubric assigns half the grade to presentation and half to implementation.
Prepare a **1:50 spoken explanation**, rehearse without reading, and use the
technical questions in [PRESENTATION.md](PRESENTATION.md) to check understanding.
A high score depends on the teacher's assessment of the actual work and delivery.

The author asked not to create a presentation. The existing outline and
rehearsal draft remain available in PRESENTATION.md. The teacher's PPTX homework
therefore remains an author task outside this implementation submission work.

The existing repository is public:
[Efeblk/event-category-classifier](https://github.com/Efeblk/event-category-classifier).
The public project includes the implementation, raw and cleaned data, licensing
and evidence described above.

| Teacher requirement | Current status |
|---|---|
| Working implementation and training scripts | Complete locally; 60 tests and fresh reproduction pass |
| Detailed README with folder/script purpose | Complete |
| Preprocessing script and documentation | Complete in prepare_data.py and data/README.md |
| Raw and cleaned datasets | Included with source attribution and preparation audits |
| Public reuse licenses | MIT scripts and CC BY 4.0 dataset, separately identified |
| Benchmark and technical explanation | Current dev comparison, frozen historical test and paper references documented |
| Solo contribution file | Guide prepared; author name, student ID and personal work details pending |
| PPTX and spoken presentation | Author task; presentation creation excluded at the author's request |
| Public GitHub submission URL | Public repository linked above |

The author deferred personal details. Complete the record described in
[contributions/README.md](contributions/README.md), add the real author name to
the final PPTX, then submit that PPTX and the public repository URL. Rehearse
without reading. Do not invent a team, claim agent-written changes as unaided
work or label paper reference scores as measured project results.
