# Human-input robustness findings

Three agents independently tested the model, HTTP API and browser, then reviewed
each other's components. The implementation fixes below address observed failures.
No new dependency, neural training or paid provider call was introduced.

## Inputs and reproducible method

The human-written inputs are news excerpts from the
[original PTC-SemEval20 release](https://zenodo.org/records/3952415), CC BY 4.0,
with [creator attribution and terms](data/SOURCE.md). This is corpus-based testing,
not a recruited user study. Ordinary conversation and malformed inputs are
separate synthetic controls.

The standard-library runner uses behavioral probes inspired by
[Ribeiro et al.'s CheckList methodology](https://aclanthology.org/2020.acl-main.442/).
It samples one annotated span from each of 25 distinct official dev articles,
seed 42. These article IDs do not overlap the original labeled training articles.
The pinned archive's SHA256 is checked before opening it. Only whitelisted
regular dev articles and their **unlabeled** TC template are read.

Each excerpt has seven variant cases: whitespace, uppercasing, apostrophe style,
one transposed-word typo, an emoji suffix, omitted context and unrelated context.
The report has **25 originals + 175 generated cases = 200 diagnostic cases**,
plus six synthetic controls. Twenty-three apostrophe cases are no-ops, leaving
152 actually modified pairs. Fractions exclude no-op pairs.

Run after preparation/training:

```powershell
python stress_test.py
```

[evidence/robustness.json](evidence/robustness.json) preserves each original source,
offsets, source/member hashes, model provenance, transformations and predictions.
The same command repeated produces identical report bytes. There is no fitting,
technique gold or accuracy calculation. These examined inputs are no longer
eligible for a fresh final evaluation.

## Fixed gaps

| Observed failure | Implemented behavior |
|---|---|
| Unknown excerpt plus familiar context received a technique | Excerpt lexical coverage is required; context and numeric counts cannot substitute |
| Editing while a request was pending could show the old result or annotation | Edits/selections invalidate the request generation; late successes/errors cannot overwrite current state |
| Example tooltip exposed its technique through the record ID before guessing | Tooltips use display numbers; annotation appears after comparison |
| Invalid Unicode or hidden-only text reached inference | Shared Python/API validation rejects surrogates, unsupported controls and format/mark-only inputs; genuine emoji and combining text remain supported |
| Duplicate JSON fields, nonfinite numbers or ambiguous HTTP bodies were accepted | Strict UTF-8 JSON, unique fields, fixed byte lengths/content type, nesting checks and a five-second read timeout |
| Cached examples/scores from another model run could appear current | Data/code hashes, seed, scope, context margin and selected models must match |

Input text and warnings render through text nodes. Literal HTML is displayed
without execution. Limits and highlights count Unicode code points, including
emoji. No request bodies or credentials are logged.

## Remaining model sensitivities

The table counts label changes among actually changed source pairs. Changes are
not automatically classification errors: these articles lack technique gold.

| Modification | Changed pairs | NB label changes | Hybrid LR | SVM |
|---|---:|---:|---:|---:|
| Whitespace | 25 | 0 | 0 | 0 |
| Uppercase | 25 | 0 | 7 | 0 |
| Apostrophe style | 2 | 0 | 0 | 0 |
| One typo | 25 | 3 | 6 | 8 |
| Emoji suffix | 25 | 0 | 4 | 0 |
| No context | 25 | 0 | 6 | 2 |
| Unrelated context | 25 | 0 | 6 | 2 |

Hybrid LR includes capitalization and text-shape counts, so lowercased lexical
features do not make its scores invariant to case or added symbols. A result
unchanged on two apostrophe pairs is very limited evidence.

All three methods now abstain on unknown/emoji-only excerpts with familiar
context. But `Hello, can you help me with this?` still receives technique labels.
Latin-script Turkish, `2026` and `...` can also receive labels in the character
models. The dataset has no neutral training class, and the models have no language
detector or calibrated out-of-domain rejection system. The demo states these
limits and warns on missing context/no-letter input. A rejection threshold or
neutral detector would need representative labeled data and separate evaluation.

## Verification

The full **79-test** unittest suite covers input/API/model regressions using fixtures and
injected provider responses. Browser checks cover stale success/error responses,
button recovery, answer concealment, literal HTML, emoji/whitespace highlighting,
Unicode limits, guesses, keyboard submission and a 375-pixel viewport. Actual
preparation/training and a fresh temporary reproduction pass; the original three
cleaned benchmark files remain unchanged. The selected model still has **57.35%
dev accuracy / 0.3957 macro F1**. A fresh labeled final test remains pending.
