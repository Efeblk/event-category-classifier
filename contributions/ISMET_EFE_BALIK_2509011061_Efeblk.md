# İsmet Efe Balık (2509011061, GitHub: Efeblk)

Solo project. I did all parts:

- Chose the task: slot filling (city, date, event name) for event-search requests.
- Wrote the preprocessing for Google SGD (Events) and Amazon MASSIVE (English), including slot mapping, BIO tagging, deduplication and leak removal.
- Built and checked the 50-request messy test set.
- Trained and compared the dictionary baseline, Naive Bayes, Logistic Regression, Linear SVM, BERT, RoBERTa and BERTweet.
- Ran the training-data experiments (dataset mixing, typo augmentation, AI-written training data) and the learning curve.
- Checked and corrected the labels of the Codex test set.
- Added Jev as a benchmark for event type and ticket count.
- Built the local demo that compares a simple parser, the best classic model and BERTweet side by side, and shows the results as charts.

I used an AI coding assistant (Claude) to help write and review code.
