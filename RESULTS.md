# Results

These scores use public Gametime event titles. The labels are the ticket site's categories and were not reviewed by hand.

## Training

Seed: 42. 3,506 titles. Split: 2,103 training, 701 validation, 702 test examples. One performer's titles never cross partitions. The vocabulary is fitted on training data only. Validation selects the model.

Each learned method runs with two feature sets. **Words** uses single words and adjacent word pairs. **Characters** uses groups of 3 to 5 characters.

| Model | Features | Validation macro F1 | 5-fold CV macro F1 | Test macro F1 | Test accuracy |
|---|---|---:|---:|---:|---:|
| Majority baseline | - | 0.2823 | 0.282 ± 0.000 | 0.2821 | 73.36% |
| Naive Bayes | Words | 0.3905 | 0.386 ± 0.030 | 0.4280 | 75.93% |
| Naive Bayes | Characters | 0.3616 | 0.367 ± 0.034 | 0.3853 | 75.21% |
| Logistic Regression | Words | 0.5377 | 0.561 ± 0.019 | 0.5424 | 70.37% |
| **Logistic Regression** | **Characters** | **0.5616** | **0.546 ± 0.039** | **0.5765** | **68.38%** |
| Linear SVM | Words | 0.5588 | 0.543 ± 0.033 | 0.5567 | 74.36% |
| Linear SVM | Characters | 0.5406 | 0.518 ± 0.025 | 0.5529 | 71.51% |

Validation selects Logistic Regression with character features. The saved model retains its training fit.

- **Accuracy misleads here.** 73% of test titles are concerts, so always answering `concert` scores 73.36% accuracy but only 0.28 macro F1. Macro F1 gives each category equal weight, so it is the metric used for selection.
- **Naive Bayes stays close to the baseline.** It mostly predicts the large concert class. Logistic Regression and Linear SVM use balanced class weights and recover the smaller classes.
- **Words and characters are close.** Cross-validation repeats training on 5 grouped folds of the 2,804 train and validation rows. It ranks Logistic Regression with words slightly first (0.561), and the top four models are within one standard deviation of each other.

## Error analysis

The selected model makes 222 errors on the 702 test titles ([errors.csv](evidence/errors.csv)).

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| concert | 0.818 | 0.750 | 0.782 |
| theatre | 0.576 | 0.514 | 0.543 |
| stand_up | **0.341** | 0.496 | **0.404** |

- **Names carry no category.** 158 errors are concert ↔ stand-up, mostly bare names such as "Bob Dylan" → stand_up and "Anthony Jeselnik" → concert. A text model cannot know who a performer is unless the training data contains that performer, and the grouped split prevents that.
- **Theatre is broad.** 57 errors are theatre ↔ concert. Well-known shows without a category word fail too: "Wicked" and "Othello" → concert. Gametime's theater category also includes ballet, circus, variety shows, and some orchestra concerts.
- **Label noise exists.** For example, "St. Louis Symphony Orchestra - Live at The Sheldon" is labeled theatre.
- Only 15 of the 222 error titles contain an explicit category word.

## Same-input comparison

36 titles, 12 per class, sampled from the test split with seed 42 ([method_comparison.json](evidence/method_comparison.json)). They were not used to select the model or write rules. A method that answers `unclear` is counted as wrong.

| Method | Correct | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Parser | 7/36 | 19.4% | 0.291 |
| Logistic Regression | 23/36 | 63.9% | 0.648 |
| Jev | Not evaluated | - | - |

The parser answers only when a title contains an explicit word ("Musical", "Ballet", "Philharmonic", "Comedy Club"). It answers `unclear` for bare names. Its request rules also misfire on titles: "That's Not Me Comedy Tour" reads as an exclusion of comedy. Jev is the method that could use knowledge of performers; a complete run would fill its row.

## Transfer to requests

The title-trained model was also run on the 40 inspected Turkish and English requests ([request_transfer.json](evidence/request_transfer.json)). These are not a blind benchmark.

| Method | Correct | Accuracy |
|---|---:|---:|
| Parser | 24/40 | 60.0% |
| Logistic Regression | 12/40 | 30.0% |

The parser's rules were written for requests and handle Turkish and negation. Logistic Regression learned English titles, so it fails on requests. A model needs training data that matches its input.

Run `train.py` and `compare_methods.py` to create detailed reports in `reports/`. Checked-in snapshots are in [evidence/](evidence/). Jev tests use mocked responses and provide no measured Jev accuracy.

Before submission: add the author's name and set repository access for the teacher.
