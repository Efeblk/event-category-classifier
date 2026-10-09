# Measured results

The current upgrade uses **train/dev only** from our custom article-grouped
split of the original SemEval-2020 Task 11 labeled training corpus, seed 42.
It fits 3,795 annotation rows from 231 articles and evaluates on 1,048 dev rows
from 54 articles. Development is model-selection data.

## Current development benchmark

Every vocabulary, numeric scaler and classifier fits train alone. Dev macro F1
selects the overall winner and each method's best variant.

| Candidate | Dev accuracy / micro F1 | Dev macro F1 |
|---|---:|---:|
| Majority | 36.26% | 0.0380 |
| NB, word | 42.46% | 0.0941 |
| NB, character | 42.84% | 0.1126 |
| LR, word | 44.75% | 0.2945 |
| LR, character | 50.10% | 0.3264 |
| SVM, word | 44.08% | 0.2737 |
| SVM, character | 50.10% | 0.3068 |
| LR, excerpt + wider context | 50.10% | 0.3410 |
| SVM, excerpt + wider context | 48.09% | 0.3114 |
| **LR, words + characters + wider context + structure** | **57.35%** | **0.3957** |

The selected model is `logistic_regression_hybrid_structure`.
Against the original selected model's **dev** accuracy of 49.24%, the gain is
**8.11 percentage points**; dev macro F1 rises from 0.3324 to 0.3957.
Against the same wider-context word LR candidate above, the gain is 7.25 points.

**Final test evaluation is pending.** Original test errors informed the upgrade,
and exploratory train/dev comparisons of regularization, context and features
informed the final configuration. Current reports contain no test predictions
and every candidate's test metric is null. These dev gains do not establish an
increase in test accuracy. A fresh independently labeled evaluation is needed.

## Explainable feature upgrade

Normalization uses English NFKC/case/whitespace and retains punctuation.
The selected LR concatenates four independent feature groups:

| Group | Representation | Weight |
|---|---|---:|
| Excerpt words | TF-IDF word 1-2 grams | 1.0 |
| Context words | Separate TF-IDF word 1-2 grams | 1.0 |
| Excerpt characters | TF-IDF char_wb 3-5 grams | 0.6 |
| Text structure | 22 numeric counts, DictVectorizer and train-fitted MaxAbsScaler | 0.5 |

Context contains the excerpt plus up to 1,000 original source characters on each
side, with whitespace normalized afterward. The train/dev maximum is 2,799
normalized characters. Exact original excerpt boundaries remain unchanged.

Numeric features measure log length, phrase length categories, mean word length,
word diversity and repetition, capitalization, digits, punctuation, modal words,
standalone negation words and exact excerpt occurrences in context. They use no
gold label or article ID. The negation count is a lexical cue, not a complete
negation parser. Exact repetition does not detect paraphrases.

TF-IDF uses min_df=2 and sublinear term frequency. NB has alpha=1 smoothing;
LR uses C=1, balanced class weights and max_iter=2000; Linear SVM uses C=1,
balanced class weights and max_iter=5000. There are no new dependencies,
pretrained embeddings or neural models. See [classifier.py](classifier.py).

The benchmark makes a forced choice for each annotation. Demo methods abstain
independently if all fitted **lexical** channels are zero, even if numeric text
features are present. Probabilities are not calibrated correctness guarantees.
Displayed contributions describe associations in the class score, not causes.

## Development errors

The selected model makes **447/1,048 dev errors**. Loaded language becomes
name-calling 52 times; name-calling becomes loaded language 40 times.

| Technique | Dev examples | Selected-model F1 |
|---|---:|---:|
| Loaded language | 380 | 0.6952 |
| Repetition | 105 | 0.6457 |
| Flag-waving | 47 | 0.6154 |
| Doubt | 106 | 0.6094 |
| Name-calling / labeling | 195 | 0.5691 |
| Black-and-white fallacy | 13 | 0.3784 |
| Thought-terminating cliches | 14 | 0.1600 |
| Bandwagon / reductio ad Hitlerum | 8 | 0.0000 |

The zero F1 on bandwagon shows that the improvement does not solve every class.
Rare-class estimates can change with a few errors. Macro F1 weights all 14 class
F1 values equally; accuracy weights annotation rows equally. Forced-choice
multiclass micro F1 equals accuracy.

[evidence/metrics.json](evidence/metrics.json) contains every class and confusion
count. See the [dev confusion matrix](evidence/confusion_matrix.png) and
[first 20 dev errors](evidence/errors_sample.csv), retained in dataset order.
Complete errors regenerate into ignored reports/errors.csv.

## Separate 40-case dev comparison

All methods receive the same original excerpt and wider context, sampled
uniformly without replacement with seed 42. This small sample is not an
independent final evaluation.

| Method | Correct / 40 | Accuracy |
|---|---:|---:|
| Character NB | 20 | 50.0% |
| Hybrid LR | 21 | 52.5% |
| Wider-context SVM | 21 | 52.5% |
| Jev | - | **Not evaluated; zero attempts** |

Per-case predictions and fixed-14-label macro F1 are in
[evidence/method_comparison.json](evidence/method_comparison.json).
Demo examples also come from dev without filtering by correct predictions.
The CLI rejects test samples for current dev-only artifacts.

## Frozen historical baseline

[evidence/baseline/metrics.json](evidence/baseline/metrics.json) preserves the
original nine-candidate run, including its custom test of 1,285 annotations
from 72 articles. The old context used 350 characters on each side and a
1,500-character cap. Dev macro F1 froze its choices before test scoring.

Its selected word-plus-context LR scored **47.32% test accuracy / 0.3542 macro
F1**, with 49.24% / 0.3324 on dev. Historical character SVM scored 50.04% test
accuracy / 0.3614 macro F1; it was not the dev winner. These old test scores
do not evaluate the current upgrade.

The historical run can be regenerated with `python train.py --profile baseline`,
using separate reports/baseline and artifacts/baseline destinations.
Current training never opens its test examples.

## Published benchmark context

The paper's **corrected Appendix B, Table 9** reports official technique
classification micro F1 of **63.74% for ApplicaAI** and **25.20% for its length-only
Logistic Regression baseline**.
Source: [Da San Martino et al. (2020)](https://aclanthology.org/2020.semeval-1.186/).

These are reference numbers, not reproduced scores. Official test data,
competition repeated-span matching and training differ from our experiment.
Our archive lacks official dev/test technique gold labels.

## Reproduction and limits

`python reproduce.py` rebuilds preparation and development training in a fresh
temporary directory from a separately verified cached archive. It compares
all five cleaned files, four reports, ten candidate dev prediction arrays and
saved-model metadata/dev predictions. See
[evidence/reproduction_check.json](evidence/reproduction_check.json).
Comparison timestamps and request milliseconds are not deterministic.

The original three cleaned benchmark files are byte-for-byte unchanged. Separate
development files reconstruct only train/dev sources with wider contexts.
One exact duplicate was removed from 6,129 raw annotations, giving 6,128 rows.
Forty-one spans retain distinct labels, which a single forced choice cannot
always satisfy.

Whole articles and identical complete inputs stay in one partition. Still,
122 model-normalized excerpt strings recur across the original partitions with
different contexts (116 under whitespace-only normalization). This is not a
held-out publisher evaluation, and no confidence interval is claimed.

The original authors curated the CC BY 4.0 corpus. Our validation, normalization,
duplicate audit and grouping are reproducible preprocessing, not an original
scrape or manual relabeling. A perfect data-novelty grade cannot be guaranteed.
The task classifies a **provided English fragment**; it does not locate new
propaganda spans, verify truth or determine author intent.
