# Submission checklists

This page is the handoff checklist for the course work. It turns the project brief into evidence that a reviewer can open, reproduce, and question. It does not replace the course's official rubric if the instructor publishes an updated one.

## Mid-semester: minimum defensible package

- [ ] The MVP notebook runs from a fresh runtime with no failed cells.
- [ ] The dataset source, actual event count, label convention, feature count, seed, and split counts are printed.
- [ ] The stated budget is at least 250,000 events, or the submission explains the genuine blocker and labels a smaller run as diagnostic.
- [ ] Logistic Regression, XGBoost, and a batch-normalized/dropout MLP are represented; any unavailable dependency or fallback is explicit.
- [ ] Low-level, high-level, and all-feature comparisons use the same documented split and data budget.
- [ ] Scaling/preprocessing is fitted on training rows only.
- [ ] ROC-AUC, average precision, Brier score or a calibration plot, supplemental accuracy, and runtime are saved by code.
- [ ] Model and threshold choices use validation data; the test set is not used to tune them.
- [ ] The report and demo identify simulated-data and scope limitations.
- [ ] Every figure and reported number traces to the saved run artifacts.

## End-semester: evidence of a finished benchmark

- [ ] The selected benchmark uses the declared 250k–1M event scale, or records the practical limit and actual total.
- [ ] `results.csv`, `summary.md`, configuration, split metadata, and prediction artifacts exist for the reported run.
- [ ] The final candidate is selected on validation only and evaluated once on the frozen test partition.
- [ ] Ranking, calibration, operating-threshold, and compute metrics are reported with their split names.
- [ ] At the selected threshold, include signal efficiency, background rejection, precision, F1, accuracy, and a confusion matrix.
- [ ] Optimizer comparisons (for example AdamW versus AdaGrad) are bounded, documented, and selected on validation—not on test performance.
- [ ] Any multi-seed or bootstrap results label the aggregation rule and seed list.
- [ ] The dashboard loads exported evidence without retraining and explains the result boundaries in the interface.
- [ ] Docker starts the review app with `docker compose up --build` and the health check passes.
- [ ] A teammate reproduces the documented path on a clean environment before submission.
- [ ] The final report avoids discovery or production-detector claims and distinguishes planned work, synthetic smoke checks, and real HIGGS measurements.

## Result-writing rule

Use an exact sentence pattern whenever reporting a score: “On the frozen test split of **N** simulated HIGGS events, **model/configuration** achieved **metric=value**.” Name the run identifier next to the table. If the run has not happened, leave a clear `[NOT RUN]` marker rather than inventing a plausible score.
