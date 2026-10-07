# MASSIVE Turkish v1.0

[Amazon MASSIVE](https://github.com/alexa/massive) contains human-localized virtual
assistant requests. Turkish was localized from English crowd-written SLURP
utterances; it is not a collection of Turkish production logs.
License: **CC BY 4.0**, Amazon.com Inc. See [LICENSE](LICENSE).
Paper: [FitzGerald et al. 2022](https://arxiv.org/html/2204.08582v2).

Archive: https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz
Pinned SHA256: `7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8`.
Version 1.0 matches the paper; 1.1 only adds Catalan.

`python prepare_massive.py` checks the cached/downloaded hash before extracting
only `1.0/data/tr-TR.jsonl` and `1.0/LICENSE`, using `tarfile` with `filter="data"`.
Preparation uses only the standard library. Generated JSONL, audit and archive
cache under data/ are ignored; evidence contains a deliberate audit snapshot.

| Official partition | Requests | Observed intents |
|---|---:|---:|
| train | 11,514 | 60 |
| dev | 2,033 | 59 (no audio_volume_other) |
| test | 2,974 | 59 (no cooking_query) |

The output preserves `id`, `partition`, `intent`, `scenario`, raw `text`, whitespace
`tokens` and aligned BIO `tags`. There are 60 intents across 18 scenarios and
55 slot types. `unclear` is not a dataset label. Top train slot counts include
date 1,797, place_name 1,053, event_name 996, person 861, time 792 and timeofday 234.

Turkish case normalization maps I→ı and İ→i. Tokens retain punctuation and suffixes.
De-annotating must reproduce the raw text exactly. A slot boundary inside a
whitespace word fails and is recorded in the audit; the pinned dataset has zero
such boundaries and zero text mismatches. Similar time expressions can use
inconsistent span boundaries/types; no labels are corrected or silently expanded.

241 normalized texts appear in multiple partitions; 36 normalized texts have
multiple intents. Keep all official rows so the paper's test remains comparable.
No performer groups, random resplitting or deduplication. Tiny classes, including
four cooking_query training requests, explain relaxed class-count checks.

The old Gametime data preparation and AI-written challenge are removed. The
author will collect and label 30–50 real-person requests as a separate transfer
test; no replacement has been fabricated. The outside-repo inspection copy was
deleted after verified preparation worked.
