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
.\.venv\Scripts\python.exe compare_methods.py
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:8011](http://127.0.0.1:8011).

The demo shows three results for the same original input: a rule-based category parser, Logistic Regression, and optional Jev. It also shows the saved training results for all six local candidate models.

Use `.venv/bin/python` on Linux or macOS.

A fresh clone can generate the public sample corpus. The process needs no parent project, provider text, database, or paid API.

Run one guarded command-line prediction:

```powershell
.\.venv\Scripts\python.exe predict.py --text "Canlı caz dinlemek istiyorum"
```

The `predict.py` command retains the earlier guarded policy. The browser comparison keeps rules separate from learned classification. Load only a model that you trained locally. A joblib file can execute code when Python loads it.

## Same input, three approaches

| Method | Input and behavior |
|---|---|
| Rule-based parser | Reads explicit activity terms and applies the existing exclusion and unsupported-activity rules |
| Logistic Regression | Reads the original text with learned character features. No extra language rules apply. Zero-vocabulary input returns `unclear` |
| Jev | Reads the original text and one fixed four-label question through the TypeSafe API |

Here, parser means the specific small rule-based category recognizer. The comparison does not cover every possible parser design. Each method uses the same labels. The comparison currently needs a Logistic Regression artifact.

`compare_methods.py` runs the same 40 inspected regression cases for every method. Local evaluation gives 57.5% accuracy for the parser and 62.5% for Logistic Regression. Jev has not been evaluated. No missing or failed Jev result becomes an `unclear` prediction. A partial provider run has no comparable headline accuracy.

The preserved local run is in [evidence/method_comparison.json](evidence/method_comparison.json). It records the dataset and source-code hashes plus every case and prediction. Jev was not requested in that run.

The 40-case comparison and the 77-row training test use different datasets. Do not compare their scores as if they used the same test set. Both use generated data. Neither establishes real-user accuracy.

## Optional Jev comparison

No API call runs on page load or during default evaluation. Select the Jev checkbox to include one external API attempt in an interactive comparison.

Copy `.env.example` to `.env`. Add your TypeSafe key in that local file. Keep it out of chat and Git. Set `JEV_MAX_CALLS` from 1 to 50. Leave it at 0 to disable calls. Restart `app.py` after a settings change.

The pinned model is `jev-1.13.0`. The call limit is shared by the browser and evaluation script. Attempt receipts in `reports/jev_calls/` survive restarts. Failed calls count too. There are no automatic retries. Jev input is limited to 1,000 characters. Local methods accept up to 5,000 characters.

To evaluate all 40 cases with Jev, configure at least 40 remaining calls, then run:

```powershell
.\.venv\Scripts\python.exe compare_methods.py --include-jev
```

A limit of 50 allows this 40-case run and up to 10 interactive attempts. Do not delete the receipts to reset an experiment budget. Receipts retain input hashes, the fixed prompt version, returned model, token usage, and validated provider answers. They contain no API key or submitted text. The evaluation report retains the labeled test cases and predictions.

This is an external model comparison. It does not replace the course requirement for Naive Bayes and another class method. See the current [TypeSafe API](https://docs.typesafe.ai/api), [Choice](https://docs.typesafe.ai/primitives/choice), and [model/pricing](https://docs.typesafe.ai/models) documentation. Real Jev calls have not run in this repository. Provider tests use controlled responses.

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
| `method_comparison.py` | Compare rules and learned classification with an optional bounded Jev call |
| `compare_methods.py` | Evaluate all three approaches on the same labeled cases |
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

The current local run passes 51 tests. The provider tests make no real API calls.

The training comparison table matches all six saved results. The three-method browser check covers eight Turkish and English samples. Desktop and narrow-screen checks pass. Jev is marked as not evaluated when it is unavailable.

The course wording for a benchmark is ambiguous. This project includes an internal majority baseline. Confirm whether the teacher also requires an external published benchmark.

Method references: [scikit-learn text features](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) and [grouped evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
