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
| Synthetic | 942 messy requests written by an AI model (Claude Haiku) | MIT | train only |
| Codex test | 60 requests written by a different AI agent (OpenAI Codex), checked by hand | MIT | test only |

The preprocessing scripts download the raw data, map the slots to `city` / `date` /
`event_name`, convert character spans to BIO tags, remove duplicates, and drop dev/test
sentences that also appear in an earlier split. Counts and dropped rows are in
`data/clean/stats.json` and `data/clean/massive_stats.json`. Licence details are in
`data/LICENSE.md`.

Training sets for the experiments: **A** SGD, **B** SGD + MASSIVE, **C** SGD + typo copies,
**D** SGD + MASSIVE + typo copies (each word of 4+ letters gets one random letter edit with probability 0.15),
**E** set D + the synthetic rows. Synthetic rows that mention an event name of the messy test set are dropped (58 of
1,000), so the messy test does not leak into training. **F** (distillation experiment only) adds a chat-style copy
of every SGD, MASSIVE and synthetic sentence to set E: lowercased, common words swapped for slang ("tomorrow" →
"tmrw"), plus typos.

The Codex test set checks set E fairly: the same AI model (Haiku) helped draft both the synthetic rows and the messy
test, so a gain on the messy test alone could come from shared style. The Codex requests come from a different model.
9 of its 60 labels were changed by hand to follow the training data's rules (times such as "after 9" are not dates,
"celtics vs heat" is one event name); the original file is not kept.

## Methods

- **Dictionary baseline**: tags any span it saw in training (longest match).
- **Naive Bayes, Logistic Regression, Linear SVM**: one prediction per word from hand-made
  features (the word, its prefixes and suffixes, its shape, neighbouring words).
- **BERT, RoBERTa, BERTweet**: fine-tuned pre-trained transformers (`bert-base-uncased`,
  `roberta-base`, `vinai/bertweet-base`), trained for up to 4 epochs; BERTweet also on set E.
- **Distillation**: a trained teacher gives a probability for every tag of every training word
  (soft labels). The student learns from both the gold tags and the soft labels:
  loss = α · cross-entropy(gold tags) + (1 − α) · T² · KL(teacher, student), with T = 2 and α
  chosen on dev from 0.3, 0.5 and 0.7. Two setups: RoBERTa-large teaching BERTweet (same job,
  bigger teacher), and BERTweet teaching BERT-mini (`google/bert_uncased_L-4_H-256_A-4`, 12×
  smaller).
- **Shrinking BERTweet**: start from the trained BERTweet (set E), keep only some of its 12
  layers (6 = every other; 4 spread out = 1, 4, 7, 10; 4 bottom = 1–4), and train again with
  its own soft labels. Then store the weights as 16-bit floats or 8-bit integers (`quantize.py`).
- **Jev** (TypeSafe, optional, paid): answers what word tagging cannot, namely the event type
  and the number of tickets ("with my girlfriend" means two).

Every setting is chosen on the dev sets (SGD dev + MASSIVE dev); each test set is scored once.
The metric is exact-span F1.

## Results

Span F1. Single run with seed 42.

| Model | Training set | Chosen on dev | SGD test | MASSIVE test | Messy test | Codex test |
|---|---|---|---|---|---|---|
| Dictionary | D | longest match | 0.72 | 0.76 | 0.58 | 0.58 |
| Naive Bayes | D | alpha = 1.0 | 0.66 | 0.58 | 0.55 | 0.59 |
| Logistic Regression | D | C = 10 | 0.88 | 0.84 | 0.62 | 0.61 |
| Linear SVM | D | C = 0.1 | 0.87 | 0.84 | 0.67 | 0.66 |
| BERT | D | epoch 3 | 0.97 | 0.91 | 0.71 | 0.72 |
| RoBERTa | D | epoch 3 | 0.98 | 0.91 | 0.75 | 0.80 |
| BERTweet | D | epoch 2 | 0.98 | 0.90 | 0.78 | 0.84 |
| Naive Bayes | E | alpha = 1.0 | 0.67 | 0.58 | 0.63 | 0.71 |
| Logistic Regression | E | C = 3 | 0.89 | 0.83 | 0.87 | 0.80 |
| Linear SVM | E | C = 1 | 0.89 | 0.83 | 0.85 | 0.81 |
| **BERTweet** | E | epoch 4 | **0.98** | 0.91 | **0.95** | **0.95** |

Distillation (training set E unless noted; BERT-mini trained for 8 epochs):

| Student (size) | Teacher | Labels | Dev | MASSIVE test | Messy test | Codex test |
|---|---|---|---|---|---|---|
| RoBERTa-large (355M) | none | gold | 0.967 | 0.92 | 0.93 | 0.91 |
| BERTweet (135M) | none | gold | 0.966 | 0.91 | 0.95 | 0.95 |
| BERTweet (135M) | RoBERTa-large | gold + soft, α = 0.3 | 0.966 | 0.91 | 0.94 | 0.95 |
| BERT-mini (11M) | none | gold | 0.957 | 0.89 | 0.84 | 0.83 |
| BERT-mini (11M) | BERTweet | gold + soft, α = 0.5 | 0.960 | 0.89 | 0.86 | 0.83 |
| BERT-mini (11M), set F | none | gold | 0.956 | 0.90 | 0.84 | 0.85 |
| BERT-mini (11M), set F | BERTweet | gold + soft, α = 0.7 | 0.957 | 0.91 | 0.85 | 0.86 |

Shrinking BERTweet (set E, soft labels from the full BERTweet, α chosen on dev). Size is the stored weights;
scores from `quantize.py`, so the 32-bit rows are full-precision runs and can differ slightly from training,
which scores with bf16 (≈ sizes: parameters × 4 bytes):

| Model | Layers kept | Stored as | Size | Dev | Messy test | Codex test |
|---|---|---|---|---|---|---|
| BERTweet | all 12 | 32-bit | 537 MB | 0.966 | 0.95 | 0.95 |
| BERTweet | all 12 | 16-bit | 269 MB | 0.966 | 0.95 | 0.95 |
| BERTweet | all 12 | 8-bit | 137 MB | 0.944 | 0.88 | 0.90 |
| BERTweet, α = 0.7 | 6 (every other) | 16-bit | **184 MB** | 0.964 | **0.95** | **0.95** |
| BERTweet, α = 0.7 | 6 (every other) | 8-bit | 93 MB | 0.954 | 0.92 | 0.89 |
| BERTweet, α = 0.3 | 4 bottom (1–4) | 16-bit | 155 MB | 0.965 | 0.90 | 0.94 |
| BERTweet, α = 0.3 | 4 bottom (1–4) | 8-bit | 79 MB | 0.956 | 0.87 | 0.92 |
| BERTweet, α = 0.5 | 4 spread (1, 4, 7, 10) | 32-bit (training) | ≈ 310 MB | 0.964 | 0.89 | 0.90 |
| BERT-mini, α = 0.5 | — | 32-bit (training) | ≈ 45 MB | 0.960 | 0.86 | 0.83 |

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
5. **AI-written messy training data helps a lot, and the gain holds on the independent Codex
   test**: BERTweet goes from 0.84 to 0.95 there, Linear SVM from 0.66 to 0.81. The clean test
   scores stay the same.
6. **Distillation only helps with a stronger teacher.** RoBERTa-large is 3× bigger but ties
   BERTweet on dev and is worse on messy text, so it had nothing to teach: the distilled
   BERTweet stayed within 0.01 of plain BERTweet. Jev was a weaker teacher still (0.73 span F1
   on 300 dev sentences, against 0.96 for BERTweet). When BERTweet teaches the 12× smaller
   BERT-mini, the soft labels raised dev F1 for every α tried (0.957 → 0.958–0.960) and messy
   F1 by about 0.01–0.02; the messy gains are within noise.
7. **A 12× smaller model keeps the clean-text score but not the messy one**: BERT-mini reaches
   0.960 dev F1 (BERTweet 0.966) but only 0.86 on messy requests (BERTweet 0.95). Adding 9k
   chat-style copies (set F) did not close the gap. Dev is clean text, so choosing α on dev
   cannot reward gains on messy text.
8. **Shrinking BERTweet itself works much better than starting from a small model.** Half of
   BERTweet (6 layers) stored in 16-bit is 184 MB, 2.9× smaller than the full model, with the
   same messy and Codex F1 (0.95). Cut-down BERTweet keeps its knowledge of casual text, which
   BERT-mini never had. Keeping the bottom 4 layers beat keeping 4 spread-out layers on the
   Codex test (0.94 vs 0.90): neighbouring layers already work together. 16-bit storage cost
   nothing; 8-bit costs 0.03–0.07 on messy text, so it is not worth it here. The word table
   (49M of BERTweet's 135M parameters) is why cutting layers alone shrinks the model only 1.5×.

Limitations: one seed; the messy and Codex test sets have only 50 and 60 requests, so
differences of a few points are within noise; both were written by AI models, not by real
users. Remaining BERTweet errors on the Codex test include places that are not cities
("msg", "bway") tagged as cities, and missed time-of-day words ("this evening").

Full numbers, per-slot scores and the learning curve are in `results/`.

## Folders and scripts

| Path | What it is |
|---|---|
| `prepare_sgd.py` | Downloads SGD, keeps Events dialogues, writes `data/raw/sgd/` and `data/clean/{train,dev,test}.jsonl` |
| `prepare_massive.py` | Downloads MASSIVE en-US, maps slots, writes `data/raw/massive/` and `data/clean/massive_*.jsonl` |
| `prepare_synthetic.py` | Converts `data/synthetic/` to BIO tags, drops rows that leak messy-test event names, writes `data/clean/synthetic_*` |
| `slots.py` | Shared helpers: reading data, BIO tags, span F1, word features, typo copies, dictionary baseline |
| `train_classic.py` | Trains and compares the dictionary, Naive Bayes, Logistic Regression and Linear SVM on sets A–E; learning curve |
| `train_bert.py` | Fine-tunes one transformer (`--model`) on set D, E or F (`--train-set`), optionally distilled from a trained teacher (`--teacher`, `--alpha`); needs a CUDA GPU |
| `jev.py` | Jev client with a hard call limit and a receipt for every call |
| `quantize.py` | Stores trained transformers as 16-bit floats and 8-bit integers and rescores them; writes `results/quantized_results.json` |
| `jev_soft_labels.py` | Asks Jev for the tag probabilities of every word (paid; capped with `--max-calls`) and scores them against BERTweet |
| `extract.py` | Loads the trained models, tags a request, and lists the follow-up questions an assistant would ask |
| `app.py`, `demo.html` | Local demo: compares all models side by side on any request, and shows the results as charts |
| `data/raw/` | Downloaded data with its licence files |
| `data/clean/` | BIO-tagged training, dev and test data, plus statistics |
| `data/messy/` | The 50-request messy test set |
| `data/synthetic/` | The AI-written training requests with their slot spans, before conversion |
| `data/codex/` | The 60-request Codex test set |
| `results/` | Scores of every model (JSON), written by the training scripts |
| `tests/` | Unit tests (no network or GPU needed) |
| `contributions/` | Contribution statement |

## How to run

Python 3.13.

```bash
python -m pip install -r requirements-lock.txt
python prepare_sgd.py          # uses data/raw if present, otherwise downloads
python prepare_massive.py
python prepare_synthetic.py
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
python train_bert.py --model vinai/bertweet-base --train-set E
python train_bert.py --model roberta-large --train-set E          # teacher for BERTweet
python train_bert.py --model vinai/bertweet-base --train-set E --teacher artifacts/roberta_large_E --alpha 0.3
python train_bert.py --model google/bert_uncased_L-4_H-256_A-4 --train-set E --tag bert_mini_8ep --epochs 8
python train_bert.py --model google/bert_uncased_L-4_H-256_A-4 --train-set E --tag bert_mini_8ep --epochs 8 --teacher artifacts/bertweet_E --alpha 0.5
```

The other distillation runs use the same commands with `--alpha 0.3` / `0.5` / `0.7` and `--train-set F`.
Shrinking (also with `--alpha 0.3` / `0.5` / `0.7`, and `--keep-layers 0,3,6,9` or `0,1,2,3` with tags
`bertweet_4L` / `bertweet_4L_bottom`), then the precision check:

```bash
python train_bert.py --model vinai/bertweet-base --train-set E --init-from artifacts/bertweet_E \
    --keep-layers 0,2,4,6,8,10 --teacher artifacts/bertweet_E --alpha 0.7 --tag bertweet_6L
python quantize.py artifacts/bertweet_E artifacts/bertweet_6L_E_kd0.7 artifacts/bertweet_4L_bottom_E_kd0.3
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
| No overlap between splits | `clean_split` in `slots.py` drops dev/test sentences seen in an earlier split; counts in `data/clean/*stats.json`. Synthetic rows that mention messy-test event names are dropped (`data/clean/synthetic_stats.json`) |
| Table numbers | `results/classic_results.json` (rows for sets D and E) and `results/{bert,roberta,bertweet,bertweet_E,roberta_large_E}_results.json`; distillation: `results/*_kd<alpha>_results.json` next to the gold-label run of the same student (the README shows the α with the best dev F1); Jev as teacher: `results/jev_teacher_check.json` (310 calls, about $0.04); shrinking: `results/bertweet_{6L,4L,4L_bottom}_E_kd*_results.json` (layers, parameters) and `results/quantized_results.json` (sizes, scores per precision) |
| Classic numbers are reproducible | `python train_classic.py` rebuilds `results/classic_results.json` on a CPU |
| Code works | `python -m unittest discover -s tests` (also run by GitHub Actions) |

## Licence

Code: MIT (`LICENSE`). Data keeps its source licences (`data/LICENSE.md`).
