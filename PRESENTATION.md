# Solo presentation: Spot the Manipulation

Target **1:50**, including slide changes. Rehearse from ideas without reading.
The latest rubric rewards finishing under two minutes.

## Five-slide outline

| Slide | Time | Visible content |
|---|---|---|
| 1. Spot the Manipulation | 0:00-0:15 | "Either support us or destroy the country." / Black-and-white fallacy / Invented illustration |
| 2. Data and preparation | 0:15-0:35 | 14 techniques; 6,128 examples; CC BY 4.0; validate offsets, normalize, audit duplicates |
| 3. Models and upgrade | 0:35-1:05 | NB / LR / SVM; TF-IDF; words + characters + wider context + 22 structure counts; train 3,795 / dev 1,048 |
| 4. Measured dev results | 1:05-1:35 | Dev accuracy 49.24% -> 57.35%; macro F1 0.3324 -> 0.3957; final test pending |
| 5. Reproducible comparison | 1:35-1:50 | Fresh rebuild matches ten candidates; guess-before-reveal demo; supplied excerpt classification; Jev not evaluated |

Keep **development results; fresh final test pending** visible on the results
slide. The historical 47.32% test result belongs to the old pipeline, not the
upgrade. Paper references belong in Q&A notes. Preload a dev excerpt if using
the demo; a wrong prediction can illustrate limitations.

## Spoken rehearsal draft

"Either support us or destroy the country." This hides alternatives behind two
choices: a black-and-white fallacy. My project classifies the persuasion technique
in a selected news excerpt using 14 categories.

I use the SemEval corpus under Creative Commons. My script reconstructs 6,128
examples from character offsets, validates boundaries, normalizes whitespace and
removes one exact duplicate. Different annotations on the same span remain.
The source authors had already curated the corpus.

I compare Naive Bayes, Logistic Regression and Linear SVM with TF-IDF, which
reduces the weight of common words. Naive Bayes assumes features are independent
given the class; LR learns weighted class evidence; SVM learns separating margins.

Short phrases miss context. The upgrade combines excerpt words, character
patterns, wider surrounding text and 22 simple counts, including length,
punctuation and exact repetition. It keeps ordinary Logistic Regression.

Whole articles stay together. Every vocabulary and scaler fits train only.
Development macro F1 selects the winner and gives rare techniques equal weight.

Dev accuracy rises from 49.24 to 57.35 percent; macro F1 rises from 0.3324 to
0.3957. These are selection results. The original test errors informed the
upgrade, so a fresh final evaluation is still needed.

A fresh rebuild matches all ten candidates. The demo lets you guess before
revealing independent predictions. This classifies provided fragments; it does
not verify their truth.

## Technical questions to rehearse

- **Why macro F1?** Loaded language has 380/1,048 dev annotations. Macro F1
  averages precision/recall-based F1 for all 14 techniques equally.
- **Is 57.35% test accuracy?** No, it is dev accuracy. The original model scored
  49.24% on dev and 47.32% on its custom test. Old test error analysis helped
  design the upgrade; a fresh final evaluation is pending.
- **Why not evaluate again on the same test?** Once test findings influence
  development, that set cannot independently validate the resulting choices.
- **How is TF-IDF computed?** Sublinear TF is 1 + log(count). Smoothed IDF is
  log((1+n)/(1+df)) + 1, followed by L2 normalization. All statistics fit train.
- **What does NB assume?** Conditional feature independence given the class.
  Language violates it; n-grams capture some local combinations.
  Multinomial NB accepts nonnegative TF-IDF as weighted feature counts.
- **What does LR learn?** Per-class weights and intercepts with softmax
  probabilities. C=1 regularization limits weights; balanced class weights
  reduce majority dominance. Probabilities are not calibrated guarantees.
- **What does SVM learn?** Linear separating boundaries with squared hinge loss.
  It uses balanced class weights and has no probability output here.
- **Why character features?** Spelling patterns generalize across related forms;
  they do not themselves understand argument logic.
- **How are features combined?** FeatureUnion concatenates excerpt word TF-IDF,
  context word TF-IDF, character TF-IDF and scaled numeric counts. Their weights
  are 1, 1, 0.6 and 0.5. LR learns the resulting class coefficients.
- **What are the numeric counts?** Length, punctuation, capitals, digits, word
  diversity/repetition and a few lexical cues. Exact phrase occurrences in
  context help repetition, but miss paraphrases. No label or article ID is a feature.
- **How is context represented?** Up to 1,000 source characters on either side,
  including the excerpt, normalized afterward. It is a separate word channel.
- **What changed in the data?** Separate development files use only original
  train/dev articles. Original benchmark files, labels and partitions are unchanged.
- **Are splits official?** No. The archive has technique gold only for official
  training articles. Our split groups articles with seed 42.
- **Is there leakage?** Whole articles and identical complete inputs are grouped.
  122 model-normalized excerpts recur with different contexts across the original
  partitions; this limits span-only comparisons and independence claims.
- **Why multiple labels on one span?** They are distinct source annotations.
  One row per annotation differs from official repeated-span matching; one
  prediction cannot satisfy conflicting gold labels on identical inputs.
- **What did you clean?** Offset validation, whitespace representations,
  one duplicate removal and grouping. I did not scrape or relabel the source.
- **What are the published scores?** Corrected Appendix B Table 9: ApplicaAI
  63.74% micro F1; length-only LR 25.20% on official test. Different data/scoring
  make them references, not our reproduction.
  [Task paper](https://aclanthology.org/2020.semeval-1.186/).
- **What are feature chips?** Positive weighted contributions to the class score.
  NB uses log-likelihood differences from its class mean. They show associations.
- **What happens with unknown text?** Each demo model abstains if fitted lexical
  channels are zero. Numeric counts alone cannot bypass this guard.
- **What remains difficult?** Bandwagon has zero dev F1 on eight examples.
  More accuracy does not mean all techniques are solved.
- **Did you use Jev?** The optional pinned, budgeted adapter remains. No paid
  evaluation was authorized or run; its scores are absent.

## Sources and submission

Use [data/SOURCE.md](data/SOURCE.md), [data/README.md](data/README.md),
[Zenodo](https://zenodo.org/records/3952415) and [RESULTS.md](RESULTS.md).
Current dev numbers trace to [evidence/metrics.json](evidence/metrics.json);
old test results live under [evidence/baseline/](evidence/baseline/README.md).

Final PPTX text/tables must remain editable with CC BY 4.0 attribution in notes.
**No final PPTX has been exported:** the required presentation runtime is
unavailable here. The obsolete intent/slot presentation was removed; its prior
version remains in Git.

Author identity, truthful contribution filename, public GitHub publication and
submission are deferred at the author's request. Do not invent names or team
contributions. Keep development and historical test numbers clearly separated.
