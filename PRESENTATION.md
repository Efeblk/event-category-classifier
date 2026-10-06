# Three-minute presentation

Slides: [presentation.pptx](presentation.pptx). Speaker notes hold the timing for each slide. The slides still show the earlier synthetic-request version and need updating to these numbers.

## 1. Task - 30 seconds

Show three event titles: "Elf The Musical", "Metallica", "Kevin Hart". Explain the three labels: concert, theatre, stand-up. An event app needs to tag listings by category. This project classifies titles; it does not search events.

## 2. Data and methods - 60 seconds

Public data: 3,506 Gametime ticket listings from Hugging Face and Kaggle (CC BY-NC 4.0). The ticket site's categories are the labels. One performer's titles stay in one partition, so the model cannot memorize a name and see it again in the test.

Show the three approaches: written parser rules, trained Logistic Regression, and optional pretrained Jev. Training also compares Naive Bayes, Linear SVM, and a majority baseline. Explain TF-IDF as converting text into numeric features. Each method uses word features and character features.

## 3. Results - 45 seconds

Show the training chart. Validation macro F1 selects Logistic Regression with character features: 0.58 test macro F1, against 0.28 for the majority baseline. Point out that the baseline has higher accuracy (73%) because most titles are concerts. That is why macro F1 is the metric.

Error analysis: most errors are bare names, such as a singer predicted as a comedian. Text alone cannot tell who a performer is.

Show the 36-title comparison: parser 19%, Logistic Regression 64%, Jev not evaluated. Rules only work when a title has a category word.

## 4. Demo and limits - 45 seconds

Enter "Hamilton" or "Metallica" for all methods. Show the parser returning unclear and the model guessing. Explain that a pretrained model like Jev could know the performer.

Limits: English only, unreviewed labels, imbalanced classes. Requests are a different input: the title model gets 30% on 40 Turkish and English requests.
