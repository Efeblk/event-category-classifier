# Event request classifier

A small COE025 NLP project for Turkish and English requests.

The demo compares **Parser**, **Logistic Regression**, and optional **Jev** on the same input. Each returns `concert`, `theatre`, `stand_up`, or `unclear`.

- Parser uses written rules.
- Logistic Regression learns from labeled examples using TF-IDF character features. TF-IDF converts text into numbers.
- Jev is a pretrained model accessed through the TypeSafe API.

## Run

Use Python 3.13. Run from this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python prepare_requests.py
python train.py --allow-small-prototype
python compare_methods.py
python app.py
```

Open [localhost:8011](http://127.0.0.1:8011). On Linux or macOS, activate with `source .venv/bin/activate`.

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
| `prepare_requests.py` | Create the synthetic dataset |
| `classifier.py`, `train.py` | Define, train, and evaluate the local models |
| `parser.py` | Rule-based category parser |
| `jev.py` | Optional API client |
| `method_comparison.py`, `compare_methods.py` | Compare the three methods |
| `app.py`, `demo.html` | Local browser demo |
| `tests/` | Tests without real API calls |

Training compares Naive Bayes, Logistic Regression, and Linear SVM, each with word and character features, plus a majority baseline. Related translations and paraphrases stay in one partition. Validation macro F1 selects the model; test data does not select it. Macro F1 gives each category equal weight.

## Optional Jev

Copy `.env.example` to `.env`. Set `TYPESAFE_API_KEY` and `JEV_MAX_CALLS` (1-50), then restart the app. The key stays on the server. Never commit `.env`.

Select **Include Jev** in the demo, or run `python compare_methods.py --include-jev` for all 40 comparison cases. Calls can cost money. The browser and script share the limit stored in `reports/jev_calls/`; failed attempts count. There are no retries. Leave the limit at 0 to disable calls. Do not delete receipts to reset the limit.

Jev has not been evaluated live. Missing or failed calls have no accuracy score.

## Data and limits

The prototype has **364 AI-authored examples**, below the course requirement of **1,000**. The separate 40-case comparison was inspected during development; it is not a blind benchmark. Synthetic scores do not establish real-user accuracy. See [results](RESULTS.md) and [dataset details](data/README.md).

This project classifies requests only. It does not search events or extract dates, locations, or budgets. It has no dependency on the original project.

## Tests

```powershell
python -m unittest discover -s tests -v
```

[Presentation outline](PRESENTATION.md) · [TypeSafe API](https://docs.typesafe.ai/api)
