# Results: MASSIVE Turkish v1.0

Core: 60-way intent classification. Extension: exact BIO slot spans. All numbers
below come from `evidence/`. The official train/dev/test counts are 11,514/2,033/2,974.
Human-localized requests retain the dataset labels; they are not independently
hand-reviewed here. There are 241 normalized texts shared across partitions and
36 normalized texts with multiple intent labels. Both are preserved and audited.

## Intent models

Word features are TF-IDF 1–2 grams; character features are `char_wb` 3–5 grams.
Balanced class weights are the default; `_unbalanced` means no weighting.
Macro F1 averages over the fixed 60 labels, including zero for absent classes.
Dev lacks `audio_volume_other`; test lacks `cooking_query`. Train has all 60.

| Model | Dev accuracy | Dev macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| `dummy_most_frequent` | 6.44% | 0.0020 | 7.03% | 0.0022 |
| `multinomial_nb_word` | 61.39% | 0.4836 | 61.10% | 0.4850 |
| `multinomial_nb_char` | 67.04% | 0.5370 | 66.27% | 0.5305 |
| `logistic_regression_word` | 75.36% | 0.7202 | 75.12% | 0.7042 |
| `logistic_regression_char` | 81.01% | 0.7838 | 79.15% | 0.7534 |
| `linear_svm_word` | 79.34% | 0.7464 | 78.35% | 0.7211 |
| `linear_svm_char` | 83.42% | 0.8112 | 82.95% | 0.7898 |
| `logistic_regression_word_unbalanced` | 73.64% | 0.6697 | 73.84% | 0.6658 |
| `logistic_regression_char_unbalanced` | 80.28% | 0.7373 | 80.06% | 0.7259 |
| `linear_svm_word_unbalanced` | 79.73% | 0.7614 | 78.82% | 0.7419 |
| `linear_svm_char_unbalanced` | 83.57% | 0.8108 | 83.49% | 0.8054 |

**Selected: balanced character Linear SVM**, by dev macro F1 0.8112. The best LR
is also balanced characters, dev macro F1 0.7838. Balanced weights improve both LR
feature variants and narrowly improve character SVM on dev. Word SVM prefers
unbalanced weights. Unbalanced character SVM has higher test scores; this does
not change the dev-selected winner. Both saved intent models are refitted on
train alone, without dev, then each candidate is scored once on test per run.

## Published Turkish reference

| Model | Test intent accuracy | Test slot F1 | Test joint exact match |
|---|---:|---:|---:|
| Our selected SVM + selected slot LR | 82.95% | 63.70% | 52.86% |
| Paper: XLM-R base | 86.3 ±1.2% | 74.9 ±0.7% | 65.2 ±1.7% |
| Paper: mT5 encoder-only | 87.1 ±1.2% | 76.1 ±0.7% | 67.7 ±1.7% |
| Paper: mT5 text-to-text | 86.1 ±1.2% | 77.9 ±0.6% | 68.1 ±1.7% |

Quoted reference values, not reproduced: [FitzGerald et al. (2022), Appendix D,
Tables 7–9](https://arxiv.org/html/2204.08582v2#A4). The ± intervals are the paper’s
95% confidence intervals. Evaluation uses the entire official tr-TR test with
all 60 possible output intents (59 actually occur). Slot scoring requires exact
type and whitespace-token boundaries; exact match requires the intent and all
slots to be correct. The paper trains multilingual models on all 51 locales;
we train only on Turkish and select intent and slots separately. We expect to
score below the paper because sparse features lack pretraining and shared
multilingual training. This is a benchmark reference, not an equal-data comparison.

## Parser and Jev

Rules were written from training requests only, for seven clear intents.
Unknown, negated and conflicting matches abstain. No test-driven rule tuning.

| Split | Coverage | Accuracy when answered | Overall accuracy |
|---|---:|---:|---:|
| Full dev | 11.31% (230/2,033) | 82.17% | 9.30% |
| Full test | 11.70% (348/2,974) | 79.31% | 9.28% |
| 40-request test sample | 20.00% (8/40) | 87.50% | 17.50% (7/40) |

On that same fixed sample, LR scores 33/40 (82.50%), macro F1 0.8149.
Parser sample macro F1 is 0.1267, over the gold intents present in the sample.
This small uniform sample is not the full paper benchmark. **Jev: not evaluated**
(0/40); no live calls were made. A partial run deliberately has null accuracy.
The v2 prompt contains 60 short intent definitions plus `unclear`; mocked
UTF-8 requests pass the 8,000-byte cap. Choice supports up to 255 options.

## Removable slot extension

Each token has Turkish-aware word, prefix/suffix, number/capital/apostrophe,
neighbor and sentence-boundary features, encoded with `DictVectorizer`.
Token decisions are independent; an invalid `I-x` starts a `B-x` span.
Both NB variants are predeclared dev candidates. All models fit train only.

| Slot model | Dev span F1 | Test span F1 | Dev joint exact match | Test joint exact match |
|---|---:|---:|---:|---:|
| `all_O` | 0.0000 | 0.0000 | 24.45% | 25.22% |
| `multinomial_nb` | 0.5056 | 0.4978 | 42.89% | 43.14% |
| `bernoulli_nb` | 0.1752 | 0.1645 | 27.10% | 27.57% |
| `logistic_regression` | 0.6586 | 0.6370 | 54.40% | 52.86% |
| `linear_svm` | 0.6565 | 0.6384 | 55.53% | 53.70% |

**Selected slot model: Logistic Regression**, by dev span F1, even though slot SVM
is slightly better on test. Joint exact match uses the frozen selected intent SVM.

| Highlighted slot | Selected model test F1 |
|---|---:|
| `date` | 0.7795 |
| `time` | 0.5805 |
| `timeofday` | 0.6857 |
| `place_name` | 0.6372 |
| `event_name` | 0.6248 |
| `artist_name` | 0.5962 |
| `person` | 0.6467 |

Parser slot rules cover **date, time and timeofday only**: dev F1 0.6063, test F1
0.6129. These three-type results cannot be compared directly with the all-slot
F1 above. Jev slots are omitted: the documented API has no arbitrary span
extraction primitive; no free-text response parser was added.

## Error analysis and limitations

Character features outperform words for NB, LR and SVM, consistent with Turkish
suffix variation. Intent confusions include `general_quirky` → `qa_factoid` (22),
`calendar_query` → `calendar_set` (14), and the reverse (10). Broad conversational
queries overlap factual questions, while calendar words alone do not distinguish
asking from adding. `general_quirky` recall is 0.3609. See the top 15 pairs and
per-intent scores in metrics.json; the plot groups confusion into 18 scenarios.

The slot LR has 329 same-type overlapping boundary errors, 89 exact-boundary
wrong-type errors, 694 missed spans and 397 spurious spans. Counts pair type
errors before overlapping boundary errors. Whole whitespace tokens retain
suffixes: a suffix inside a token cannot become a separate span. Training labels
sometimes disagree on whether a daypart joins the time span; for example,
`sabah dokuzda` is one time span, while `öğleden sonra dörde` can split into
timeofday and time. We preserve those annotations and do not repair labels.

The audit found no inside-word boundaries or de-annotation mismatches. Official
absent classes required relaxing evaluation split checks; tiny classes also
make macro F1 sensitive. The sparse index conversion to int32 is required by
LinearSVC with the locked sklearn version and is covered by fitted-model tests.

## Reproduction and pending work

Two fresh output directories reproduced data, audits, all model metrics, errors,
confusion PNG and the comparison sample exactly. Comparison timestamps/timings
are excluded from equality; predictions, scores, statuses and code hashes match.
See `evidence/reproduction_check.json`. No trained artifacts or full generated
dataset are committed. Local tests pass; the existing CI runs the same suite,
but no remote CI run was triggered during this rewrite.

Author: collect 30–50 requests from real people as a new transfer test; approve
a future paid Jev run; ask whether the teacher accepts token-classification slots;
update the unchanged presentation.pptx, add the author name, and arrange GitHub access.
