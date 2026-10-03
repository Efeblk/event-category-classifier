# Project scope and course requirements

Project name: Turkish event category classifier.

Input: an event title and description.

Output: concert, theatre, or stand-up.

This is a topic classification problem. It can support category assignment in an event catalog. It is a separate student project. It has no runtime dependency on Bi' Plan.

## Course requirements

The supplied Week 2 deck gives the following requirements. Slide numbers include the cover slide.

| Requirement | Source | Project status |
|---|---|---|
| One classification problem | Slide 26 | Three event categories form one classification task |
| Naive Bayes and another class method | Slide 26 | Naive Bayes, Logistic Regression, and linear SVM work |
| Benchmark comparison | Slide 26 | Majority-class baseline works. Confirm whether an external benchmark is also required |
| At least 1,000 examples with clear labels | Slide 29 | 1,182 distinct examples with explicit collector labels. Human label review remains necessary |
| Check the dataset license | Slide 29 | Source terms checked. No open corpus license established. Data and models stay outside Git |
| Five students per team | Slide 26 | Add the actual team members before submission |
| GitHub repository before presentation | Slide 26 | Publish the separate code repository before the talk |
| Three-minute presentation and two-minute Q&A in Week 4 | Slide 26 | Talk outline and Q&A notes are in `PRESENTATION.md`. Create the final slides and rehearse |

## Grading

Week 2 slide 27 assigns 40% to technological depth, 10% to originality, and 50% to presentation performance.

Technological depth includes correct code, method comparison, feature engineering, data splits, metrics, and error analysis. The slide does not score line count or software architecture size. This project adds useful depth through model comparison and correct evaluation.

The individual-word versus word-pair experiment tests feature engineering. A word pair is a bigram. For example, two adjacent words form one feature. The experiment keeps this change separate from the classifier comparison.

## Acceptance checks

- Import at least 1,000 distinct labeled examples.
- Exclude conflicting event families.
- Keep training, validation, and test groups separate.
- Learn text features from training data only.
- Select the model from validation results only.
- Report every model on the same test partition.
- Save real metrics, per-class results, wrong predictions, and a confusion matrix.
- Predict all three sample categories in the local demo.
- Run the tests without paid calls or a production database.

## Before submission

Confirm the data permission and the teacher's benchmark requirement. Review a sample of labels. Record any label changes as a new dataset version. Do not edit test labels merely to improve the existing score.

Add the five team members. Prepare the final slides. Assign a short speaking part to each team member if the teacher expects all members to speak. Rehearse within three minutes.

The data importer, text features, model comparison, error review, and demonstration form useful team work areas. Each member must understand the full evaluation process.
