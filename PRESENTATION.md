# 3-minute presentation (solo)

Intent classification is Project 1’s core. Slots are one removable extension
slide. Use numbers from evidence; rehearse to 180 seconds. `presentation.pptx`
remains the old deck and must be updated by the author.

| Time | Slide | Say / show |
|---|---|---|
| 0:00–0:25 | Problem | “What does a Turkish request ask for?” Show dev request `yarın sabah altıya alarm kur` → `alarm_set`. Classifying intent does not execute the request. |
| 0:25–0:55 | Public data | MASSIVE v1.0, CC BY 4.0; human-localized requests. 11,514 train / 2,033 dev / 2,974 test. 60 intents. Keep official splits and audit duplicates. |
| 0:55–1:30 | Methods | Majority, NB, LR, SVM. Word and character TF-IDF turn text into sparse features. Character grams capture suffixes. Dev macro F1 gives small classes equal weight and selects the winner. Parser and optional pretrained Jev are independent. |
| 1:30–2:15 | Core results | Character SVM: dev 83.42% accuracy / 0.8112 F1; test 82.95% / 0.7898. Show scenario confusion and calendar-query/set confusion. Paper XLM-R Turkish intent accuracy is 86.3 ±1.2%; it has pretraining and training across 51 locales. Our demo LR test is 79.15% / 0.7534. Parser test coverage 11.70%, answered accuracy 79.31%. Jev is not evaluated. |
| 2:15–2:45 | Optional slots | Highlight `yarın` (date), `sabah altıya` (time). Explain token prefix/suffix and neighbor features, BIO tags, and simple repair. Dev-selected slot LR: test exact span F1 0.6370; joint exact match 52.86%. One wrong slot boundary makes a span wrong. This extension can be removed without changing intent classification. |
| 2:45–3:00 | Takeaway | Character features are a strong explainable baseline, below multilingual pretrained references. Next: real-person transfer requests, an approved Jev run, and teacher confirmation about slots. |

Use screenshots if a live demo risks the 3-minute limit. Demo examples are dev,
never selected test examples. The 40-request same-input sample is secondary:
Parser 7/40, LR 33/40, Jev 0 calls (not evaluated).

## Q&A (2 minutes)

- **Why NB plus LR/SVM?** They satisfy the methods taught in class and allow a
  fair feature comparison on the same official data.
- **Why not choose higher test SVM or slot SVM scores?** Selection was frozen
  using dev macro F1 / dev slot F1, before test scoring.
- **Why keep duplicates?** 241 texts cross official partitions; dropping them
  would change the benchmark. We disclose that and retain official labels.
- **Are all 60 classes in every split?** Train has all 60; dev lacks one and test
  lacks one. Fixed 60-label macro F1 includes an absent-class zero; accuracy is
  compared with the paper on the full official Turkish test.
- **Why below the paper?** Sparse Turkish-only features have no pretraining;
  the paper fine-tunes large models jointly across 51 locales. Reference numbers
  were quoted, not reproduced.
- **Are these production Turkish requests?** No: humans localized English
  crowd-written benchmark requests. A new request set from people is pending.
- **Why slots?** It is optional token classification, with separate files and
  exact span metrics. The teacher’s scope decision is pending.
- **Why no Jev score?** Paid calls require explicit approval; no calls were made.
  Its ledger counts failed attempts, has no retries, and caps the shared budget.
