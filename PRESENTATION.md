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

List the methods: majority baseline, Naive Bayes, Logistic Regression, and linear SVM. Mention word, word-bigram, and character features.

## Slide 3: Result and interpretation, 55 seconds

Title: Raw-model comparison
State that validation macro F1 selected Logistic Regression with character groups of length 3 to 5. Explain that macro F1 gives equal weight to each label.

Show the selected model's test macro F1 of 0.8195 and test accuracy of 83.12%. State that the test set has 77 rows.

Explain that these scores measure performance on the generated corpus. They do not estimate accuracy for real users.

Mention the 40-case guard regression set. The demo with guards reached 80%. The model with its vocabulary guard reached 62.5%. The keyword baseline reached 57.5%. State that the team inspected these cases during guard fixes. Do not present them as a blind benchmark.

## Slide 4: Demo and next work, 35 seconds

Title: One request in, one label out
Run one Turkish or English request. Show that the command-line tool and browser use the same guards.

State the limits. The system has four labels. It does not search events, extract request details, or keep chat memory. Its rules cannot interpret every negation.

Show the next data step. Five team members can author 200 requests each under one guide. The team must record consent, license, author code, and paraphrase groups. A separate human-reviewed set must stay unused until final evaluation.

## Q&A notes

**Why Naive Bayes?** The course requires it. It also gives a simple text-classification reference.

**Why character features?** They can handle spelling variation in Turkish and English. Validation selected this feature set.

**Why use macro F1?** Macro F1 gives each label equal weight. It combines precision and recall.

**Why group paraphrases?** Closely related wording in both training and test data can make the score too high.

**Why add `unclear`?** A runtime classifier needs a result for vague, mixed, unsupported, and non-positive requests.

**Is the regression set a benchmark?** No. We inspected it and used it to fix guard defects.

**Does the project meet the 1,000-example rule?** No. The current prototype has 364 examples. Confirm the accepted data method with the teacher.

**What is the benchmark?** The experiment includes a majority baseline. Ask the teacher whether the assignment also requires an external published result.
