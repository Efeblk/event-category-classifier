# User-request prototype results

This experiment classifies one Turkish or English request. It does not classify provider event descriptions. The old 91.98% accuracy does not apply to this task. Earlier experiments remain in [archive/](archive/README.md).

## Dataset

The corpus contains 364 AI-authored examples in 52 intent families. It contains 208 Turkish examples and 156 English examples. Each of the four labels has 91 examples. These texts are generated examples, not observed user requests. Translations and paraphrases of one intent stay in the same group.

The course requires at least 1,000 examples. The current corpus does not satisfy that requirement. A separate human evaluation is also needed before a real-user accuracy claim.

| Partition | Examples | Families |
|---|---:|---:|
| train | 210 | 30 |
| validation | 77 | 11 |
| test | 77 | 11 |

## Raw model comparison

The vocabulary comes from training data only. Validation macro F1 selects the model. The saved model keeps its training fit. Each model uses the same group partitions and seed 42. These scores use the generated corpus. They exclude the demo input rules.

| Model | Validation macro F1 | Test macro F1 | Test accuracy |
|---|---:|---:|---:|
| dummy_most_frequent | 0.1071 | 0.0769 | 18.18% |
| multinomial_nb | 0.6386 | 0.6227 | 63.64% |
| logistic_regression_unigram | 0.6747 | 0.6965 | 70.13% |
| logistic_regression_bigram | 0.6771 | 0.6812 | 68.83% |
| logistic_regression_char | 0.8185 | 0.8195 | 83.12% |
| linear_svc | 0.7199 | 0.6772 | 68.83% |

The selected model is `logistic_regression_char`. Character features cover short letter sequences within words. They can share evidence across spelling variants and word endings. Their selection used validation results. The first word-feature prototype remains preserved. This is an iterative experiment, not a claim that all development choices were independent of previously inspected results.

![Raw model confusion matrix](evidence/confusion_matrix.png)

## Separate regression examples

The separate file contains 40 AI-authored examples. It contains no exact normalized training text. We inspected it while fixing input rules. It is now a regression set. It is not a blind external benchmark.

| Decision method | Correct regression examples | Accuracy |
|---|---:|---:|
| model_with_vocabulary_guard | 25/40 | 62.5% |
| demo_with_input_guards | 32/40 | 80.0% |
| keyword_baseline | 23/40 | 57.5% |

The input rules check explicit exclusions, multiple positive activity categories, music-player commands, and unsupported movie requests. The learned model handles the remaining input. Rules cannot cover every negation or meaning. No score threshold claims calibrated confidence.

## Error review

The first prototype confused some unseen wording with the unclear category. Character features reduced that problem on the fixed corpus split. The final regression set still contains wrong predictions. Errors include indirect activity descriptions, new vocabulary, and difficult category boundaries. Do not treat these synthetic results as accuracy on real users.

All six visible demo samples returned their expected categories in the T3 browser. The checks cover Turkish and English requests and ambiguous requests. These are demonstration checks, not a benchmark.

## Evidence

[Raw model metrics](evidence/metrics.json) preserve the dataset hash, package versions, family counts, per-class scores, and code hashes. [Regression results](evidence/request_challenge.json) preserve every generated case, prediction, and rule reason. The trained model and generated CSV stay local. `prepare_requests.py` rebuilds the corpus from the checked-in examples.

Dataset SHA-256: `ad21d7eaa6673f27f95c69f54b54cfec2924fceadec1bf9c4ab267c96d5b836a`.

The remaining course gates are the accepted 1,000-example dataset, benchmark interpretation, team names, final slides, and rehearsal.
