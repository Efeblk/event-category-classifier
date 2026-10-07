# Event category classifier

A small COE025 NLP project. It classifies event listing titles as `concert`, `theatre`, or `stand_up`.

The data is public: Gametime ticket listings published by Rebrowser on [Hugging Face](https://huggingface.co/datasets/rebrowser/gametime-dataset) and [Kaggle](https://www.kaggle.com/datasets/rebrowser/gametime-dataset) under CC BY-NC 4.0. The labels are the ticket site's own categories.

The demo compares **Parser**, **Logistic Regression**, and optional **Jev** on the same input.

- Parser uses written rules.
- Logistic Regression learns from labeled titles using TF-IDF features. TF-IDF converts text into numbers.
- Jev is a pretrained model accessed through the TypeSafe API.

A method returns `unclear` when it cannot choose. `unclear` is not a training class, and it counts as wrong in scoring.

## Run

Use Python 3.13. Run from this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python prepare_events.py
python train.py
python compare_methods.py
python app.py
```

`prepare_events.py` downloads about 7 MB from Hugging Face at a pinned commit and writes `data/events.csv`. Open [localhost:8011](http://127.0.0.1:8011). On Linux or macOS, activate with `source .venv/bin/activate`.

## Demo requests

Try these in order to illustrate the methods' strengths and limits. Parser and Logistic Regression outputs below were verified with the current model. Jev outputs are expected, not live-verified; check them before presenting.

| Level | Request | Parser | Logistic Regression | Expected Jev |
|---|---|---|---|---|
| Easy | Bu akşam bir konsere gitmek istiyorum. | `concert` | `concert` | `concert` |
| Medium | Bu gece sahnede blues çalan birilerini dinleyelim. | `unclear` | `concert` | `concert` |
| Hard | Müzik olmasın, mikrofon başında şaka yapan biri olsun. | `unclear` | `unclear` | `stand_up` |

The easy request names the category explicitly. The medium request describes a music performance without the Parser's keywords. The hard request describes stand-up indirectly and excludes music. These are selected demo examples, not an accuracy benchmark.

## Project files

| File | Purpose |
|---|---|
| `prepare_events.py` | Download the public listings and build the labeled dataset |
| `classifier.py`, `train.py` | Define, train, and evaluate the local models |
| `parser.py` | Rule-based category parser |
| `jev.py` | Optional API client |
| `method_comparison.py`, `compare_methods.py` | Compare the three methods |
| `app.py`, `demo.html` | Local browser demo |
| `tests/` | Tests without network or real API calls |

Training compares Naive Bayes, Logistic Regression, and Linear SVM, each with word and character features, plus a majority baseline. Titles by the same performer stay in one partition. Validation macro F1 selects the model; test data does not select it. Macro F1 gives each category equal weight.

## Optional Jev

Copy `.env.example` to `.env`. Set `TYPESAFE_API_KEY` and `JEV_MAX_CALLS` (1-50), then restart the app. The key stays on the server. Never commit `.env`.

Select **Include Jev** in the demo, or run `python compare_methods.py --include-jev` for all 36 comparison titles. Calls can cost money. The browser and script share the limit stored in `reports/jev_calls/`; failed attempts count. There are no retries. Leave the limit at 0 to disable calls. Do not delete receipts to reset the limit.

Jev has not been evaluated live. Missing or failed calls have no accuracy score.

## Data and limits

3,506 unique English titles after cleaning. Many titles are only a performer's name, so the models partly learn names. Gametime labels were not reviewed by hand and contain some noise. See [results](RESULTS.md) and [dataset details](data/README.md).

The 40 Turkish and English requests in `data/request_challenge.json` are a secondary test: does a model trained on titles understand requests? It mostly does not.

This project classifies categories only. It does not search events or extract dates, locations, or budgets. It has no dependency on the original project.

## Tests

```powershell
python -m unittest discover -s tests -v
```

[Presentation outline](PRESENTATION.md) · [TypeSafe API](https://docs.typesafe.ai/api)
