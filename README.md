# Turkish event category classifier

COE025 Project 1. This project classifies a Turkish event title and description as `concert`, `theatre`, or `stand_up`. It uses one small part of the Bi' Plan idea. It runs independently from Bi' Plan.

The project compares Naive Bayes, Logistic Regression, and a linear Support Vector Machine (SVM). It also compares word features with word-pair features. A majority-class baseline always predicts the most common training label.

## Run on Windows

Use Python 3.13. Run these commands from this repository.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
# Read an existing, approved local snapshot. This command does not change it.
.\.venv\Scripts\python.exe prepare_data.py 'C:\path\to\events.json'
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:8011](http://127.0.0.1:8011). The model predicts one event category. It does not search a live catalog.

The prepared dataset and trained model already exist in the original local checkout. A Git clone contains the code, tests, and aggregate results. It does not contain provider text or the trained model. Read [the dataset guide](data/README.md) before importing data. The importer needs at least 1,000 distinct examples and all three labels.

For Linux or macOS, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

Run one prediction without a browser:

```powershell
.\.venv\Scripts\python.exe predict.py --text 'Canli muzik konseri ve caz orkestrası'
```

Load only a model that you made locally with `train.py`. A joblib model file can execute code when you load it.

## Experiment

TF-IDF converts text to numeric features. It gives more weight to terms that help distinguish documents. The experiment tests individual words and word pairs. It preserves Turkish letters and handles `I` and `İ` before lowercase conversion.

The importer removes repeated text and excludes event families with conflicting labels. It connects related records using titles, descriptions, production keys, and source pages. Each family stays in one data partition.

The training program uses about 60% of the data for training, 20% for validation, and 20% for testing. It learns the vocabulary from training data only. It selects the model with the best validation macro F1. Macro F1 gives equal weight to each category. It evaluates the frozen models on test data after selection. It saves the selected model without a second fit.

The local experiment has 1,182 unique examples in 965 families. Logistic Regression with individual words won validation selection. Its test macro F1 is **0.9168**. Its test accuracy is **91.98%**. Read [the results and error review](RESULTS.md). These scores measure agreement with collector labels. They do not prove independent human accuracy.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Read a snapshot, remove duplicates, assign groups, and write an audit |
| `classifier.py` | Normalize text and define the compared models |
| `train.py` | Split data, train models, compare results, and save the winner |
| `predict.py` | Classify one input from the command line |
| `app.py`, `demo.html` | Show a local browser demonstration |
| `tests/` | Check grouping, normalization, model selection, and data separation |
| `PROJECT.md` | Map the course requirements to this project |
| `PRESENTATION.md` | Give a three-minute talk outline and Q&A notes |

`train.py` writes metrics, an error list, a confusion matrix, and split IDs to `reports/`. It saves the model to `artifacts/model.joblib`. Git ignores these directories. [RESULTS.md](RESULTS.md) preserves the aggregate result and dataset hash.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use small controlled data. They make no external API calls. The real-data check runs only when the local CSV exists. GitHub CI runs the controlled checks.

The code has no paid API, database, collector, or deployment requirement. Model probabilities are not calibrated confidence. The model always predicts one of the three labels, including for unrelated text.

The course asks for benchmark comparison. This project includes an internal baseline. Confirm whether the teacher also requires a published external benchmark. Confirm data reuse permission before submission or data sharing. The checked provider terms do not give this project an open data license. See [the source terms](data/README.md#source-terms).

Method references: [scikit-learn text features](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) and [grouped evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
