# Slot filling for event-search requests

COE025 Natural Language Processing, Project 1: Text Classification.
Solo project by İsmet Efe Balık (see `contributions/`).

A user types a request like *"wanna catch the lakers game w my dad tmrw"*. The task is to
classify every word as part of a **city**, a **date**, an **event name**, or nothing (BIO
tags), then read off the slots: event = `lakers`, date = `tmrw`. This is word-level text
classification (slot filling), which is not one of the lecture-slide tasks.

The goal is robustness: models trained on clean assistant data must also work on messy,
lowercase, typo-filled requests.

## Data

| Set | Source | Licence | Train / dev / test sentences |
|---|---|---|---|
| SGD Events | Google Schema-Guided Dialogue, user turns of the Events services | CC BY-SA 4.0 | 5,586 / 626 / 784 |
| MASSIVE en-US | Amazon MASSIVE, rows with date, place or event slots | CC BY 4.0 | 2,607 / 488 / 633 |
| Messy test | 50 casual requests with slang and typos (AI-drafted, checked and corrected by hand) | MIT | test only |

The preprocessing scripts download the raw data, map the slots to `city` / `date` /
`event_name`, convert character spans to BIO tags, remove duplicates, and drop dev/test
sentences that also appear in an earlier split. Counts and dropped rows are in
`data/clean/stats.json` and `data/clean/massive_stats.json`. Licence details are in
`data/LICENSE.md`.

Training sets for the experiments: **A** SGD, **B** SGD + MASSIVE, **C** SGD + typo copies,
**D** SGD + MASSIVE + typo copies (each word of 4+ letters gets one random letter edit with probability 0.15).

## Methods

- **Dictionary baseline**: tags any span it saw in training (longest match).
- **Naive Bayes, Logistic Regression, Linear SVM**: one prediction per word from hand-made
  features (the word, its prefixes and suffixes, its shape, neighbouring words).
- **BERT, RoBERTa, BERTweet**: fine-tuned pre-trained transformers (`bert-base-uncased`,
  `roberta-base`, `vinai/bertweet-base`), trained on set D for 4 epochs.
- **Jev** (TypeSafe, optional, paid): answers what word tagging cannot, namely the event type
  and the number of tickets ("with my girlfriend" means two).

Every setting is chosen on the dev sets (SGD dev + MASSIVE dev); each test set is scored once.
The metric is exact-span F1.

## Results

Span F1. All rows are trained on set D. Single run with seed 42.

| Model | Chosen on dev | SGD test | MASSIVE test | Messy test | Messy event names |
|---|---|---|---|---|---|
| Dictionary | longest match | 0.72 | 0.76 | 0.58 | 0.00 |
| Naive Bayes | alpha = 1.0 | 0.66 | 0.58 | 0.55 | 0.40 |
| Logistic Regression | C = 10 | 0.88 | 0.84 | 0.62 | 0.18 |
| Linear SVM | C = 0.1 | 0.87 | 0.84 | 0.67 | 0.19 |
| BERT | epoch 3 | 0.97 | 0.91 | 0.71 | 0.40 |
| RoBERTa | epoch 3 | 0.98 | 0.91 | 0.75 | 0.46 |
| **BERTweet** | epoch 2 | **0.98** | 0.90 | **0.78** | **0.69** |

Main findings:

1. **Every model loses accuracy on messy requests.** The classic models find cities well but
   miss casual dates ("tmrw") and lowercase event names ("taylor swift").
2. **Mixing MASSIVE and typo copies into training helps the classic models on messy text**
   (Linear SVM: 0.52 on set A, 0.67 on set D).
3. **BERTweet, pre-trained on tweets, handles messy text best.** It had the best dev score of
   the three transformers, so it is the main model. Its event-name F1 on messy requests is
   0.69 against 0.19 for Linear SVM.
4. **More data of the same kind barely helps**: Linear SVM trained on 10% of SGD already
   reaches 0.90 test F1, and on all of it 0.92.

Limitations: one seed; the messy test set has only 50 requests, so differences of a few
points are within noise; the messy requests were drafted with AI help.

Full numbers, per-slot scores and the learning curve are in `results/`.

## Folders and scripts

| Path | What it is |
|---|---|
| `prepare_sgd.py` | Downloads SGD, keeps Events dialogues, writes `data/raw/sgd/` and `data/clean/{train,dev,test}.jsonl` |
| `prepare_massive.py` | Downloads MASSIVE en-US, maps slots, writes `data/raw/massive/` and `data/clean/massive_*.jsonl` |
| `slots.py` | Shared helpers: reading data, BIO tags, span F1, word features, typo copies, dictionary baseline |
| `train_classic.py` | Trains and compares the dictionary, Naive Bayes, Logistic Regression and Linear SVM on sets A–D; learning curve |
| `train_bert.py` | Fine-tunes one transformer (`--model`) on set D (needs a CUDA GPU) |
| `jev.py` | Jev client with a hard call limit and a receipt for every call |
| `extract.py` | Loads the trained models, tags a request, and lists the follow-up questions an assistant would ask |
| `app.py`, `demo.html` | Local demo: compares all models side by side on any request, and shows the results as charts |
| `data/raw/` | Downloaded data with its licence files |
| `data/clean/` | BIO-tagged training, dev and test data, plus statistics |
| `data/messy/` | The 50-request messy test set |
| `results/` | Scores of every model (JSON), written by the training scripts |
| `tests/` | Unit tests (no network or GPU needed) |
| `contributions/` | Contribution statement |

## How to run

Python 3.13.

```bash
python -m pip install -r requirements-lock.txt
python prepare_sgd.py          # uses data/raw if present, otherwise downloads
python prepare_massive.py
python train_classic.py        # about 5 minutes on a CPU
python -m unittest discover -s tests
```

Transformers (CUDA GPU; about 2 minutes each on an RTX 4070 SUPER):

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
python -m pip install -r requirements-gpu.txt
python train_bert.py --model bert-base-uncased
python train_bert.py --model roberta-base
python train_bert.py --model vinai/bertweet-base
```

Demo (http://127.0.0.1:8000). Jev is only called when `.env` has a `TYPESAFE_API_KEY` and a
`JEV_MAX_CALLS` above 0 (see `.env.example`):

```bash
python app.py
```

## How to check the results

Each claim above can be checked without a GPU:

| Claim | Where to look |
|---|---|
| Rubric items | Preprocessing: `prepare_*.py`. Training: `train_*.py`. Raw and clean data: `data/`. Licence: `LICENSE`. Contributions: `contributions/` |
| Naive Bayes plus taught methods (LR, SVM, BERT), and benchmarks | `train_classic.py`, `train_bert.py`; dictionary baseline and Jev as benchmarks |
| No test data used for choosing settings | `train_classic.py` and `train_bert.py` pick settings and epochs by dev F1 only |
| No overlap between splits | `clean_split` in `slots.py` drops dev/test sentences seen in an earlier split; counts in `data/clean/*stats.json` |
| Table numbers | `results/classic_results.json` (rows for set D) and `results/{bert,roberta,bertweet}_results.json` |
| Classic numbers are reproducible | `python train_classic.py` rebuilds `results/classic_results.json` on a CPU |
| Code works | `python -m unittest discover -s tests` (also run by GitHub Actions) |

## Licence

Code: MIT (`LICENSE`). Data keeps its source licences (`data/LICENSE.md`).
