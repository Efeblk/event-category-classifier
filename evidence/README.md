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
| `baseline/` | Frozen original nine-candidate test benchmark and reproduction | See baseline/README.md |

Original data files and partitions are unchanged. Separate wider-context
development data contains no test rows. The old test's errors informed the
upgrade, so final test evaluation is pending. Dev results are selection results,
not independent generalization evidence. See [RESULTS.md](../RESULTS.md).

Raw/cleaned data already live under data/. Trained joblib files, all prediction
arrays, full errors and archive cache remain ignored and regenerate locally.
The root code license is MIT; dataset material retains
[CC BY 4.0 terms and attribution](../data/SOURCE.md).
