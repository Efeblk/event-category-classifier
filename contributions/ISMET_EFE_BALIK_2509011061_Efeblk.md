# İsmet Efe Balık (2509011061, GitHub: Efeblk)

Solo project. I did all parts:

- Chose the task: slot filling (city, date, event name) for event-search requests.
- Wrote the preprocessing for Google SGD (Events) and Amazon MASSIVE (English), including slot mapping, BIO tagging, deduplication and leak removal.
- Built the project's own messy-request dataset: had Claude Haiku write 2,400 requests with their slots, wrote the script that checks and cleans them (1,957 kept), and built and checked the 50-request messy test set.
- Checked and corrected the labels of the 260-request Codex test set.
- Trained and compared the dictionary baseline, Naive Bayes, Logistic Regression, Linear SVM, BERT, RoBERTa and BERTweet.
- Ran the training-data experiments (dataset mixing, typo augmentation, AI-written training data), the learning curve and the data curve, with three seeds for the main results.
- Tried distillation (RoBERTa-large → BERTweet, BERTweet → BERT-mini, Jev as a teacher) and shrinking BERTweet (fewer layers, 16-bit and 8-bit weights).
- Added bootstrap confidence intervals and confusion matrices for the main models.
- Added Jev as a benchmark for event type and ticket count, and the follow-up questions built from the found slots.
- Built the local demo that compares a simple parser, the best classic model and BERTweet side by side, and shows the results as charts.

I used an AI coding assistant (Claude) to help write and review code.
