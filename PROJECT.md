# Project scope and course requirements

Project name: Turkish and English event-request classifier.
Input: one user request.
Output: `concert`, `theatre`, `stand_up`, or `unclear`.

This project studies one text-classification task. It has no runtime dependency on Bi' Plan.

## Current scope

The classifier processes one query. It does not search an event catalog. It does not extract dates, budgets, or locations. It does not keep conversation state.

The `unclear` label covers requests without one supported positive activity. This includes vague requests, mixed target categories, unsupported activities, and pure exclusions. Explicit rules handle some music-player commands, movie requests, mixed categories, and negation. These rules do not cover all language forms.
## Course requirements

| Requirement | Current status |
|---|---|
| One classification problem | Complete. The task classifies one request into four labels |
| Naive Bayes and another method | Complete. The comparison includes Naive Bayes, Logistic Regression, and linear SVM |
| Benchmark comparison | An internal majority baseline exists. Confirm whether the teacher requires an external benchmark |
| At least 1,000 labeled examples | Open. The generated prototype has 364 examples |
| Dataset license | The original synthetic examples use CC0-1.0 |
| Five students per team | Add the actual team members before submission |
| GitHub repository | The project has a separate private remote repository. Confirm public or submission access |
| Three-minute presentation | The outline is in `PRESENTATION.md` |

The 364 examples do not complete the 1,000-example requirement. They do not represent observed users. Ask the teacher whether a documented synthetic study is acceptable before claiming course compliance.
## Experiment design

The prototype has 91 examples for each label. It has 208 Turkish and 156 English examples. The examples form 52 paraphrase families.

The split uses 210 rows for training, 77 for validation, and 77 for testing. All related Turkish and English examples stay in one partition. The training partition alone defines the TF-IDF vocabulary.

The experiment compares a majority dummy, Naive Bayes, three Logistic Regression feature sets, and linear SVM. The validation result selects character-based Logistic Regression. Its raw-model test macro F1 is 0.8195. Its test accuracy is 83.12%.

The 40-case regression set tests the demo guards and language boundaries. We used the same cases to find and fix guard defects. Therefore, this set is not a blind evaluation set. No real-user accuracy is established.

## Plan for 1,000 examples

1. Confirm with the teacher whether human-authored synthetic requests meet the requirement.
2. Ask five team members to author 200 requests each under one written label guide.
3. Record consent, source, license, author code, language, label, and paraphrase-family ID.
4. Keep translations and paraphrases of one intent in the same family.
5. Review mixed, vague, negated, and unsupported requests with a second team member.
6. Resolve disagreements before training. Preserve the original decisions in an audit file.
7. Reserve a human-reviewed set before model or guard changes. Do not use it to fix defects.
8. Report generated and observed data separately if the team later collects consented real requests.

This plan gives 1,000 documented examples without pretending that generated text came from real users. A teacher-approved synthetic corpus can meet the assignment only after the teacher confirms that interpretation.

## Submission checks

- Run the full unit test suite.
- Generate the corpus from a fresh clone.
- Train with `--allow-small-prototype` only while the corpus stays below 1,000 rows.
- Record the exact data hash and split IDs.
- Show all candidate methods on the same test partition.
- Keep raw-model metrics separate from guarded demo regression results.
- Add the five team members and repository access details.
- Rehearse the talk within three minutes.
