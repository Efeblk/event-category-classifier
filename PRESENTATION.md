# Three-minute presentation outline

Use four slides. Keep the tables small. Use the local demo only if time remains.
## Slide 1: Task and scope, 35 seconds

Title: Classifying event requests
Show one Turkish request and one English request. Show the four outputs: concert, theatre, stand-up, and unclear.

Explain that the input is a user request. The earlier event-description experiment is archived. State that this version classifies one query and does not search for events.

Explain `unclear` with two examples. Use a mixed request and a movie request.

## Slide 2: Data and methods, 55 seconds

Title: Reproducible bilingual prototype
Show these corpus counts: 364 examples, 52 intent families, 208 Turkish examples, and 156 English examples. Each label has 91 examples.

State that the examples are AI-authored. They are not observed user requests. State that the course requirement for 1,000 examples remains open.

Explain that related Turkish and English phrasings stay in one partition. Show the split: 210 training rows, 77 validation rows, and 77 test rows.

Introduce the three main approaches: a rule-based category parser, trained Logistic Regression, and optional pretrained Jev. Explain that the course training comparison also includes Naive Bayes, a majority baseline, and linear SVM. Mention word and character features.

## Slide 3: Result and interpretation, 55 seconds

Title: Same-case comparison and limits
Show the 40-case table: parser 57.5%, Logistic Regression 62.5%, and Jev not evaluated. Replace the Jev entry only after an actual complete evaluation. All methods use the same original inputs and four labels.

State that the cases were inspected during rule changes. This is a regression comparison, not a blind benchmark or real-user accuracy estimate.

Mention the separate training experiment. Validation macro F1 selected character-based Logistic Regression. Its 77-row test accuracy is 83.12%. This is a different dataset from the 40-case comparison. Macro F1 gives each label equal weight.

## Slide 4: Demo and next work, 35 seconds

Title: Same input, three approaches
Run one Turkish or English request. Show the rule-based parser and Logistic Regression results side by side. Show Jev only after an actual configured API call. Keep an unavailable Jev result marked as not evaluated.

Use "Konser değil, tiyatro istiyorum" to show an exclusion. Explain that Logistic Regression has no extra language rules in this comparison. Its earlier guarded command-line demonstration is a different policy.

State the limits. The system has four labels. It does not search events, extract request details, or keep chat memory. Its rules cannot interpret every negation.

Show the next data step. Five team members can author 200 requests each under one guide. The team must record consent, license, author code, and paraphrase groups. A separate human-reviewed set must stay unused until final evaluation.

## Q&A notes

**Why Naive Bayes?** The course requires it. It also gives a simple text-classification reference.

**Does Jev replace the class methods?** No. It is an optional external comparison. The experiment still trains and compares Naive Bayes and the other class methods.

**What does parser mean here?** It means our small rule-based category recognizer. It does not represent every parser architecture.

**How do we make the comparison fair?** All three methods receive the same original request and select from the same four labels. Learned classification has no extra language rules. Failed provider calls remain failures. A human-reviewed blind set is still required for stronger conclusions.

**Why character features?** They can handle spelling variation in Turkish and English. Validation selected this feature set.

**Why use macro F1?** Macro F1 gives each label equal weight. It combines precision and recall.

**Why group paraphrases?** Closely related wording in both training and test data can make the score too high.

**Why add `unclear`?** A runtime classifier needs a result for vague, mixed, unsupported, and non-positive requests.

**Is the regression set a benchmark?** No. We inspected it and used it to fix guard defects.

**Does the project meet the 1,000-example rule?** No. The current prototype has 364 examples. Confirm the accepted data method with the teacher.

**What is the benchmark?** The experiment includes a majority baseline. Ask the teacher whether the assignment also requires an external published result.
