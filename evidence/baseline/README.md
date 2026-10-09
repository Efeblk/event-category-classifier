# Frozen original benchmark

These five files were copied from the original nine-candidate run **before**
the feature/context upgrade. They preserve its custom article-grouped test
of 1,285 annotation rows and its original 350-character context margin.

The dev-selected word-plus-context LR scored 49.24% dev accuracy / 0.3324 macro
F1 and 47.32% test accuracy / 0.3542 macro F1. The 40-case method comparison here
is the original **test** sample. Its timestamps are not deterministic.

Reproduction passed for the original three cleaned files, four reports, nine
candidate test prediction arrays and saved-model metadata/test predictions.
These schema-1 reports are historical evidence, not current artifact inputs.
Current schema-2 artifacts require explicit evaluation scope/context provenance.

To rerun the original nine candidates on unchanged original data, use
`python train.py --profile baseline`. Generated reports/artifacts go to separate
baseline subdirectories. Current code hashes/schema metadata will differ from
the saved historical report; original candidate configurations and data remain.

These results do not evaluate the new hybrid model. See
[the current results](../../RESULTS.md) for dev results and the pending final test.
