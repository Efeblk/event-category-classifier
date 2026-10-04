# Results

These scores use synthetic requests. They do not measure accuracy on real users.

## Training

Seed: 42. Split: 210 training, 77 validation, 77 test examples. Paraphrase families do not cross partitions. The vocabulary is fitted on training data only. Validation selects the model.

Each learned method runs with two feature sets. **Words** uses single words and adjacent word pairs. **Characters** uses groups of 3 to 5 characters.

| Model | Features | Validation macro F1 | Test macro F1 | Test accuracy |
|---|---|---:|---:|---:|
| Majority baseline | - | 0.1071 | 0.0769 | 18.18% |
| Naive Bayes | Words | 0.6386 | 0.6227 | 63.64% |
| Naive Bayes | Characters | 0.7269 | 0.7508 | 77.92% |
| Logistic Regression | Words | 0.6771 | 0.6812 | 68.83% |
| **Logistic Regression** | **Characters** | **0.8185** | **0.8195** | **83.12%** |
| Linear SVM | Words | 0.7199 | 0.6772 | 68.83% |
| Linear SVM | Characters | 0.7970 | 0.8052 | 81.82% |

Validation selects Logistic Regression with character features. The saved model retains its training fit.

Character features beat word features for every method, by 0.08 to 0.14 macro F1. Turkish adds suffixes to words (`konser`, `konsere`, `konserine`), and the data has typos (`consert`, `teatr`, `standap`). Character groups still match these forms, but whole words do not.

## Error analysis

The selected model makes 13 errors on the 77 test examples ([errors.csv](evidence/errors.csv)).

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| concert | 0.706 | 0.857 | 0.774 |
| theatre | 0.826 | 0.905 | 0.864 |
| stand_up | 0.913 | 1.000 | 0.955 |
| unclear | 0.857 | **0.571** | 0.686 |

- **Negation is the main weakness.** 7 errors are exclusion-only requests such as "I do not want a concert" and "Tiyatro olmasın". The category word is present, so the model predicts that category. Bag-of-features models do not represent negation.
- 2 more `unclear` errors are near-topic: "Sinemada film izlemek istiyorum" → theatre and "Biraz eğlenmek istiyoruz" → concert.
- 4 errors are real categories with no direct keyword, such as "Suggest an event where a rapper is on stage" → theatre.

The rule-based parser gets all 9 `unclear` errors right but all 4 category errors wrong. Rules handle negation, and the learned model handles paraphrases and typos. The errors are complementary, but this project keeps the methods separate and does not combine them.

## Same-input comparison

This uses 40 separately authored cases. They were inspected during development and are not a blind benchmark. This dataset differs from the 77-example test above.

| Method | Correct | Accuracy |
|---|---:|---:|
| Parser | 23/40 | 57.5% |
| Logistic Regression | 25/40 | 62.5% |
| Jev | Not evaluated | - |

The parser and learned model run separately. No parser rules override Logistic Regression. If the input has no fitted text features, the learned classifier returns `unclear`.

Run `train.py` and `compare_methods.py` to create detailed reports in `reports/`. Checked-in snapshots are in [evidence/](evidence/). Jev tests use mocked responses and provide no measured Jev accuracy.

Before submission: reach the accepted 1,000-example dataset, confirm the teacher's benchmark requirement, add the author's name, and set repository access for the teacher.
