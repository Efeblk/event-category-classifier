# MASSIVE Turkish v1.0

Amazon MASSIVE contains crowd-written requests localized into Turkish (tr-TR).
Source: https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz
License: CC BY 4.0, Amazon.com Inc.; see LICENSE.

`python prepare_massive.py` verifies the pinned SHA256 before extracting only the
Turkish JSONL and license with tarfile's data filter. Official partitions remain
unchanged: 11,514 train, 2,033 dev, 2,974 test; 60 intents and 55 slot types.
Whitespace tokens keep Turkish suffixes. BIO alignment requires exact text and
whole-token boundaries; violations fail with an audit. Cross-partition normalized
duplicates are reported, never dropped. Generated data and archives are ignored.
