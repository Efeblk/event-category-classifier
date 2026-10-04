# Results

These scores use synthetic requests. They do not measure accuracy on real users.

## Training

Seed: 42. Split: 210 training, 77 validation, 77 test examples. Paraphrase families do not cross partitions. The vocabulary is fitted on training data only. Validation selects the model.

| Model | Validation macro F1 | Test macro F1 | Test accuracy |
|---|---:|---:|---:|
| Majority baseline | 0.1071 | 0.0769 | 18.18% |
| Naive Bayes | 0.6386 | 0.6227 | 63.64% |
| Logistic Regression | 0.8185 | 0.8195 | 83.12% |

Logistic Regression uses character groups of length 3-5. The saved model retains its training fit.

## Same-input comparison

This uses 40 separately authored cases. They were inspected during development and are not a blind benchmark. This dataset differs from the 77-example test above.

| Method | Correct | Accuracy |
|---|---:|---:|
| Parser | 23/40 | 57.5% |
| Logistic Regression | 25/40 | 62.5% |
| Jev | Not evaluated | - |

The parser and learned model run separately. No parser rules override Logistic Regression. If the input has no fitted text features, the learned classifier returns `unclear`.

Run `train.py` and `compare_methods.py` to create detailed reports in `reports/`. Checked-in snapshots are in [evidence/](evidence/). Jev tests use mocked responses and provide no measured Jev accuracy.

Before submission: reach the accepted 1,000-example dataset, confirm the teacher's benchmark requirement, add team names, and set repository access for the teacher.
