# Solo contribution record

The author deferred their name and student ID. GitHub publication uses the
existing Efeblk account, but the author must confirm the username for submission.
There are no additional team members.

Before submission, create exactly one personal file here named:

`<STUDENT_NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md`

Replace spaces in the student's name with underscores. Do not submit this guide
as the required personal contribution file. The teacher also checks GitHub diffs,
so the record must describe actual work and acknowledge assistance honestly.

## Record to complete

```markdown
# Contribution: <student name>

- Student ID: <student ID>
- GitHub username: <username>
- Role: solo project author

## My decisions and work

Describe the problem choice, data preparation decisions, implementation work,
experiments and verification you personally performed. Link the relevant files
and commits. Replace this instruction with concrete details you can explain.

## Assistance

This project used Codex and collaborating coding agents to implement and review
the preparation pipeline, classical models, evaluation scripts, tests, local
demo and documentation. Explain which changes you inspected, understood or
modified yourself. Do not claim agent-written code as unaided work.

## Verification and understanding

Record the commands you personally ran and their outcomes. Explain train-only
feature fitting, article grouping, dev macro F1 selection, the four feature
groups and why 57.35% is a development result. State any checks still pending.

## Presentation

Record your slide revisions, rehearsal timing and preparation for technical
questions. Add the final presentation and submission details when complete.
```

## Evidence available for the record

| Work | Files to inspect and understand |
|---|---|
| Source verification, offset validation, duplicate audit and grouped partitions | `prepare_data.py`, `data/README.md`, `data/cleaned/audit.json` |
| Words, characters, context and 22 generic structural counts | `classifier.py` |
| Train-only fitting, fixed-label metrics and dev-only selection | `train.py`, `RESULTS.md`, `evidence/metrics.json` |
| Fresh rebuild and deterministic comparisons | `reproduce.py`, `evidence/reproduction_check.json` |
| Same-input NB/LR/SVM comparison and optional unevaluated Jev | `method_comparison.py`, `compare_methods.py`, `jev.py` |
| Local guess-and-reveal demo | `app.py`, `demo.html` |
| Validation and regression checks | `tests/` |
| Talk and technical question preparation | `PRESENTATION.md` |

The user selected a more engaging classification task, requested collaboration
and cross-review, and authorized the explainable accuracy upgrade. Those are
documented project decisions. Personal execution, review and rehearsal claims
remain for the author to supply.
