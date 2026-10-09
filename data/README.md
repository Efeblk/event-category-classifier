# Propaganda technique classification data

This project uses the **PTC-SemEval20** English news corpus from the original
[SemEval-2020 Task 11 release](https://zenodo.org/records/3952415), distributed
under **CC BY 4.0**. See [LICENSE](LICENSE) and [SOURCE.md](SOURCE.md) for attribution.
The task classifies a **provided annotated fragment** into one of 14 techniques.
It does not decide whether arbitrary text is propaganda, identify new spans,
verify factual accuracy, or establish the writer's intent.

## Files and reproduction

Run from the repository root:

```powershell
python prepare_data.py
```

Preparation uses only Python's standard library. The pinned archive URL is
`https://zenodo.org/records/3952415/files/datasets-v2.tgz?download=1` and SHA256 is
`fd2d2c358d42b2e326e7f6f148007b384f0a0ffac5ab6db21e3889313a8516bb`.
The checksum is checked **before opening the tar archive**. Only regular
training article/technique-annotation files and the source README are extracted
with `tarfile`'s `filter="data"`. No dev templates or unrelated files are used.

| Path | Purpose |
|---|---|
| `raw/articles/article<ID>.txt` | All 371 original training articles, byte-for-byte copies |
| `raw/annotations/article<ID>.task2-TC.labels` | Original four-column tab-separated annotations |
| `raw/SOURCE_README.md` | Original release documentation, including its prior corrections |
| `cleaned/techniques.jsonl` | 6,128 validated examples and custom partitions |
| `cleaned/audit.json` | Measured transformation, duplicate and leakage audit |
| `cleaned/splits.json` | Seed, every article's partition and grouping manifest |
| `cleaned/development.jsonl` | 4,843 train/dev examples with wider context for the upgraded model; no test rows |
| `cleaned/development_audit.json` | Wider-context reconstruction, source hashes and train/dev isolation audit |
| `source/datasets-v2.tgz` | Ignored archive cache; not a second committed dataset copy |

## Preprocessing decisions

Each annotation's article ID, technique name and offsets are validated before
extraction. Offsets count **Unicode code points**, start inclusive and end
exclusive, against the exact UTF-8-decoded original article. Raw bytes are read
directly so platform newline conversion cannot shift offsets. The `text` field
is the exact original substring and is never rewritten.

The derived `clean_text` collapses Unicode whitespace to single ASCII spaces.
In the original `techniques.jsonl` benchmark, `context` includes the excerpt and
up to 350 original characters on either side, with the same whitespace
normalization and a 1,500-character cap. Negation,
punctuation, case and Unicode characters are retained. There are no hand-written
label corrections, stop-word deletion or generated examples.

The source contains **6,129 annotation rows**. One identical annotation row is
removed, leaving **6,128 examples**. Its full annotation and source line are in
the audit. There are 6,087 distinct source spans; 41 spans have multiple labels.
Those differing labels remain as separate multiclass examples. Snippets that
share words but have distinct source annotations are retained.

The archive is a **curated, already annotated corpus**, not our original web
scrape. Its source README documents corrections the dataset creators made.
Our audit describes only transformations performed by our script; converting
annotations into a table is not presented as scraping or extensive messy-data
cleaning.

## Splits and limitations

The archive contains gold technique labels only for official training articles.
Its 75 official dev articles have 1,063 span templates but no technique labels;
the pinned archive contains no official test articles. This project therefore
creates a **custom split of the labeled official training data**, not an official
SemEval test reproduction.

Seed 42 shuffles sorted article groups once and assigns approximately 65/15/20%
of groups to train/dev/test. Grouping keeps whole articles together and also
joins articles with identical original content hashes or identical normalized
excerpt/context pairs. Assignment does not consult labels, model predictions
or test scores. The script fails if training lacks any of the 14 techniques.
Dev/test may lack rare labels; evaluation uses a fixed 14-label macro F1.

Of 371 original articles, 357 have technique annotations and enter the supervised
split. The 14 articles with empty annotation files remain in raw data and are
listed in the audit; they are not invented negative examples. The split has
3,795 train / 1,048 dev / 1,285 test examples across 231 / 54 / 72 articles.
For grouping only, input identity uses the model's NFKC, lowercasing and
whitespace normalization, so equivalent model inputs stay together.

Generic short snippets can recur across partitions with different contexts. The
audit reports this explicitly. Identical excerpt/context model inputs cannot
cross partitions. Article grouping reduces shared-context leakage; it is not
a held-out publisher or topic experiment.

Each JSONL record has `id`, `article_id`, original `text`, normalized `clean_text`,
`context`, official `label`, original `start`/`end`, custom `partition` and
`source_partition="official_train"`. Stable IDs combine article ID, offsets and
label. The original benchmark fits train only, selects on dev and evaluates its
frozen test afterward. Its data and the three original preparation outputs are
retained unchanged for reproducibility.

## Wider-context development dataset

The upgraded model combines word/character features, generic text-shape features
and a wider context window. `build_development_rows(data_dir="data")` reconstructs
only the existing train/dev articles listed in `splits.json`; it never opens
test article or annotation files. It returns `(rows, audit)`. `prepare()` writes
these to `development.jsonl` and `development_audit.json` after reproducing the
original benchmark outputs.

The new table keeps the **same 3,795 train and 1,048 dev examples**, stable IDs,
raw excerpt text, code point offsets, labels and partitions. Only `context`
changes: it includes the original excerpt and up to **1,000 original characters
on each side**, followed by Unicode whitespace normalization. There is no
additional context cap; the observed maximum is 2,799 characters. Context is
selected from text and offsets, not technique labels or model predictions.

The script validates original row counts, all 14 training labels, span boundaries,
and absence of shared articles or identical model-normalized excerpt/context
pairs between train and dev. The audit fingerprints the split manifest and all
570 train/dev raw source files and records zero test sources opened.

The current upgrade fits train and reports **development results only**. The
development set already informed exploration, so these numbers are not a fresh
held-out evaluation. The original 1,285 test rows remain preserved in
`techniques.jsonl`; the upgraded default training does not score them. A new
performance claim requires untouched independently labeled articles.

Inference should supply the same excerpt and similarly broad surrounding
context. Missing or shorter context changes the input distribution; wider
context is not inferred from the technique label.
