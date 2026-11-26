# Evidence snapshots

These files are deliberate outputs from actual local runs, not training inputs.
Root snapshots describe the current **development-only upgrade**.

| File | Purpose | Regeneration |
|---|---|---|
| `metrics.json` | Ten candidates, dev selection, dev class/confusion metrics; all test metrics null | `python train.py` |
| `confusion_matrix.png` | Dev confusion of the selected hybrid LR | `python train.py` |
| `errors_sample.csv` | First 20 dev errors in dataset order | First 20 data rows of reports/errors.csv |
| `method_comparison.json` | Same-input 40-case dev comparison; Jev not evaluated | `python compare_methods.py` |
| `reproduction_check.json` | Fresh preparation/training, exact bytes and dev predictions | `python reproduce.py` |
| `robustness.json` | 25 licensed human news sources, 200 original/variant cases and six controls; no technique gold or accuracy | `python stress_test.py` |
| `baseline/` | Frozen original nine-candidate test benchmark and reproduction | See baseline/README.md |
| `test_evaluation.json` | One-time, user-authorized old-test scores for the dev-only upgrade (optimistically biased; caveat inside) | `python final_test.py` |
| `test_confusion_matrix.png` | Old-test confusion of the dev-selected hybrid LR | `python final_test.py` |

Original data files and partitions are unchanged. Separate wider-context
development data contains no test rows. The old test's errors informed the
upgrade, so its single 2026-10-10 evaluation (`test_evaluation.json`) is
optimistically biased and not a fresh final test. `metrics.json` keeps every
test metric null. Dev results are selection results, not independent
generalization evidence. See [RESULTS.md](../RESULTS.md).

The robustness sources are separate official dev articles without technique
gold labels. Generated variants are not additional human examples. No-op
variants are excluded from changed-pair fractions, and these examined sources
cannot serve as a fresh final test. See [ROBUSTNESS.md](../ROBUSTNESS.md).

Raw/cleaned data already live under data/. Trained joblib files, all prediction
arrays, full errors and archive cache remain ignored and regenerate locally.
The root code license is MIT; dataset material retains
[CC BY 4.0 terms and attribution](../data/SOURCE.md).
