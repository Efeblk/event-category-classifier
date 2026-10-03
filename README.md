# Turkish and English event-request classifier

This COE025 Project 1 classifies one user request. It returns `concert`, `theatre`, `stand_up`, or `unclear`.

The input can use Turkish or English. The classifier handles one query at a time. It does not search events. It does not extract dates, budgets, or locations. It does not keep chat memory.

`unclear` covers vague requests, mixed preferences, unsupported activities, and requests with no positive category. Explicit music-player commands, movie requests, and pure category exclusions also return `unclear`. The small rule set cannot interpret every form of negation.

## Reproduce the prototype

Use Python 3.13. Run these commands from this repository on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe prepare_requests.py
.\.venv\Scripts\python.exe train.py --allow-small-prototype
.\.venv\Scripts\python.exe evaluate_requests.py
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:8011](http://127.0.0.1:8011).

Use `.venv/bin/python` on Linux or macOS.

A fresh clone can generate the public sample corpus. The process needs no parent project, provider text, database, or paid API.

Run one guarded command-line prediction:

```powershell
.\.venv\Scripts\python.exe predict.py --text "Canlı caz dinlemek istiyorum"
```

The command-line tool and browser demo use the same input guards. Load only a model that you trained locally. A joblib file can execute code when Python loads it.

## Data

`prepare_requests.py` creates 364 AI-authored examples. The corpus contains 208 Turkish examples and 156 English examples. Each of the four labels has 91 examples.

The examples form 52 intent families. A family contains different phrasings of one intent. Turkish and English versions of the same intent have one `group_id`. This rule keeps related text in one data partition.

The corpus is a student prototype. It contains no observed real-user requests. It does not satisfy the course requirement for at least 1,000 examples. The original examples use the CC0-1.0 license. See [the dataset guide](data/README.md) and [the license](data/LICENSE).

## Methods and result

TF-IDF converts text into numeric features. The experiment compares these methods:

- majority-class dummy baseline;
- Multinomial Naive Bayes with word unigrams and bigrams;
- Logistic Regression with word unigrams;
- Logistic Regression with word unigrams and bigrams;
- Logistic Regression with character groups of length 3 to 5;
- linear Support Vector Machine with word unigrams and bigrams.

The split contains 210 training rows, 77 validation rows, and 77 test rows. Families do not cross partitions. Validation macro F1 selects the model. Macro F1 gives equal weight to each label.

Validation selected character-based Logistic Regression. Its raw-model test macro F1 is **0.8195**. Its raw-model test accuracy is **83.12%**. Read [the result and limits](RESULTS.md).

The raw-model report does not include input guards. A separate 40-case AI-authored regression set produced 80% accuracy for the demo with guards, 62.5% for the model with its vocabulary guard, and 57.5% for the keyword baseline. We inspected these cases while we fixed guard defects. They are regression cases. They are not a blind benchmark.

The project has no measured accuracy on real user requests. The generated corpus and regression cases cannot establish real-user performance or course compliance.

## Files

| File | Purpose |
|---|---|
| `prepare_requests.py` | Generate the public prototype corpus and stable family groups |
| `classifier.py` | Normalize text and define the candidate models |
| `request_policy.py` | Apply explicit request-boundary guards |
| `train.py` | Split data, compare models, and save the selected model |
| `evaluate_requests.py` | Run the separate regression cases |
| `predict.py` | Classify one request with the demo policy |
| `app.py`, `demo.html` | Run the local browser demo |
| `tests/` | Check data, models, splitting, and request guards |
| `archive/event-descriptions/` | Preserve the earlier event-description experiment |
| `archive/request-prototype-v1/` | Preserve the first request-model run |

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

GitHub CI runs the controlled tests. The tests make no paid calls.

The current local run passes 37 tests. The browser check returns the expected labels for six sample requests.

The course wording for a benchmark is ambiguous. This project includes an internal majority baseline. Confirm whether the teacher also requires an external published benchmark.

Method references: [scikit-learn text features](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) and [grouped evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
