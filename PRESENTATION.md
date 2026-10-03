# Three-minute presentation outline

Use four slides. Keep the result table small. Show the local demo only if time permits. The course allows two more minutes for Q&A.

## Slide 1: Problem and data, 35 seconds

Title: Turkish event category classification

Explain that event catalogs need category labels. Show one short Turkish event description. Show the three possible outputs: concert, theatre, and stand-up.

State that this project uses one small part of an event-discovery idea. State that it runs independently.

The local dataset has 1,182 distinct text examples. Repeated sessions do not count as different text examples. Labels come from existing collector categories. These labels can contain errors.

## Slide 2: Methods and evaluation, 55 seconds

Title: Model comparison

Explain TF-IDF in one sentence: it converts terms into numeric weights that describe each document.

Compare Naive Bayes, Logistic Regression, and linear SVM. Explain that the baseline always predicts the most common category.

Explain that one experiment uses individual words. Another uses individual words and word pairs.

Show the training, validation, and test row counts: 709, 236, and 237. Explain that related event families stay together. Explain that the validation score selects the model. The test score measures the selected model after selection.

## Slide 3: Results and errors, 60 seconds

Title: Results on unseen event families

Use the table in `RESULTS.md`. Show test macro F1 for Naive Bayes, both Logistic Regression feature sets, and SVM. Also show the baseline.

State that Logistic Regression with individual words won validation selection. Its test macro F1 is 0.9168. Macro F1 gives equal weight to each category.

Explain that SVM has a slightly higher test score. We keep Logistic Regression because we chose the model before examining test scores.

Explain two observed error patterns. Short titles lack enough context. Comedy shows overlap with theatre. Some source labels also disagree with descriptions. The result measures agreement with the supplied labels.

## Slide 4: Demonstration and limits, 30 seconds

Title: Local demonstration

Open `http://127.0.0.1:8011`. Choose one sample. Run the classifier. State the predicted category.

Explain that the model supports three categories. It cannot reject unrelated text. It does not recommend events or check ticket availability.

State the remaining work: confirm dataset reuse permission and review source labels. Do not claim an open data license until you have one.

## Q&A preparation

**Why Naive Bayes?** The course requires it. It also gives a simple text-classification reference.

**Why use macro F1?** The classes have different sizes. Macro F1 gives each class equal weight. It combines precision and recall. Precision measures the fraction of predictions that are correct. Recall measures the fraction of labeled examples that the model finds.

**Why group events before splitting?** The same show can have many dates or providers. If copies appear in training and test data, the score can overstate performance on new shows.

**Why not select SVM after seeing its test result?** That would use test data for model selection. We used validation data to select Logistic Regression.

**Are probabilities confidence?** They are model probabilities. We did not calibrate them against a separate dataset.

**Does this replace Bi' Plan?** It performs one classification task. It does not perform recommendation, collection, or publication.

**What is the benchmark?** The experiment includes a majority-class baseline. The course slide does not define the required benchmark. Confirm whether the teacher also requires a published external result.

**Can the data be public?** We have not established an open data license. The code repository excludes provider text and models.
