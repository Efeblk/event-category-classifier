# Earlier experiments

`event-descriptions/` preserves the first event-description experiment. The code came from commit `2f38391ac21bdfcf5abc86c896123604bc4df7e2`. Its scores do not measure user-request classification. Provider text and the trained model remain local.

`request-prototype-v1/` preserves the first request prototype result and its failed challenge cases. It used an intermediate, uncommitted working tree. We did not preserve a full source checkpoint for that intermediate run. Do not treat it as exact-revision verification. We used the inspected challenge cases to improve explicit input rules. Later challenge results are regression evidence, not a blind external benchmark.

The current experiment is in `evidence/` and `RESULTS.md`. Earlier results stay separate.
