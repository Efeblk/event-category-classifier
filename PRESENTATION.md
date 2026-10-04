# Three-minute presentation

## 1. Task - 30 seconds

Show one Turkish request and one English request. Explain the four labels: concert, theatre, stand-up, and unclear. This project classifies requests; it does not search events.

## 2. Data and methods - 60 seconds

Explain the 364 synthetic examples and the 52 paraphrase families. Related Turkish and English examples stay in one partition. The 1,000-example course requirement remains open.

Show the three approaches: written parser rules, trained Logistic Regression, and optional pretrained Jev. Training also compares Naive Bayes, Linear SVM, and a majority baseline. Explain TF-IDF as converting text into numeric features. Each method uses word features and character features.

## 3. Results - 45 seconds

Show the training chart. Character features beat word features for every method because they handle Turkish suffixes and typos. Validation macro F1 selects Logistic Regression with character features. Its test accuracy on 77 separate examples is 83.12%. 5-fold cross-validation gives the same ranking (0.796 ± 0.078 macro F1).

Error analysis: 7 of its 13 test errors are negations such as "I do not want a concert". The parser gets those right, but it misses paraphrases that the model gets right.

Show the 40-case comparison: parser 57.5%, Logistic Regression 62.5%, Jev not evaluated. These cases were inspected during development. They do not prove real-user accuracy.

## 4. Demo and limits - 45 seconds

Enter the same request for all methods. Show a disagreement. Explain that parser rules do not override the learned model. Keep unavailable Jev results marked as not evaluated.

State the next step: teacher-approved data with at least 1,000 labeled examples and an unseen human-reviewed evaluation set.
