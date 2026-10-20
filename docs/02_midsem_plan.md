# Mid-Semester Project Plan — HiggsLens MVP

## Goal

Deliver a working, understandable and defensible prototype quickly. The mid-semester scope should prove that the team understands the problem, can obtain and validate the official dataset, can fit the required model families, and can report a first honest comparison. Do not try to build every end-semester feature now.

## Deliverable definition

By the submission date, produce:

1. Executed notebook `notebooks/01_midsem_mvp.ipynb` with no failed cells.
2. Data overview: actual source, row count, label mapping, class balance, 28 inputs, and feature groups.
3. Three feature-set experiments: low-level only, high-level only, all 28 features.
4. At least Logistic Regression, XGBoost (with a clearly disclosed fallback only if XGBoost cannot be installed), and an MLP with batch normalization/dropout.
5. ROC-AUC, average precision, Brier score/calibration, accuracy as supplemental context, train time and prediction time where measured.
6. ROC/precision-recall/calibration plots and a model comparison table.
7. `summary.md`, `results.csv`, validation/test prediction exports and recorded run metadata.
8. A 4–6 page initial report or course-required equivalent and a short demo.

## Explicit data budget

Start with **250,000 events**, which is within the assignment's recommended 250,000–1,000,000 starter range. If download, installation or hardware delays threaten the deadline, reduce the number only as a temporary diagnostic run; the submitted benchmark should meet the stated 250,000 lower bound if feasible. Record the source and number actually used. Do not label an under-sized emergency run as meeting the official budget.

The notebook reads the official UCI compressed stream for the requested row count. It may take time to download/decompress. Never commit the dataset to GitHub. If network access fails, download the official dataset from UCI manually and follow the local-file instructions in the notebook/README. Do not silently switch to an unverified smaller mirror.

## A day-by-day version (use relative days if deadlines differ)

### Day 1 — Understand and set up

- Read `docs/01_higgs_physics_for_beginners.md`.
- Create the GitHub repository and folder structure.
- Open the notebook in Colab, check Python version, install dependencies, and verify that the source is accessible.
- Record a first run's device, package versions, seed, source and data size.

**Exit gate:** imports run; no raw data or secrets are in Git.

### Day 2 — Data quality and EDA

- Run schema checks: 29 columns when label is included, 28 features after target removal, both labels present, finite values, expected names.
- Plot class count and a few low/high-level feature distributions.
- Write a short explanation of what the plots establish and what they do not establish.

**Exit gate:** dataset and labels are understood; no preprocessing is fitted to test rows.

### Day 3 — Classical baselines

- Fit Logistic Regression on the three feature groups using a `StandardScaler` fitted only on the training partition.
- Fit XGBoost on the same three groups. Tree models may not need standardization for performance, but feature/model inputs and split remain comparable.
- Record fit time and predicted scores.

**Exit gate:** six classical runs are recorded or a failed model is clearly documented; comparisons use consistent partitions.

### Day 4 — Neural baseline

- Fit the specified compact MLP on low-level features and all features. Scaling is fitted on training rows only.
- Use a fixed seed, a fixed epoch/batch configuration and a reasonable early stop/time budget if implemented.
- Save model scores for validation and test.

**Exit gate:** at least the low-level and all-feature MLP runs finish and probabilities are finite.

### Day 5 — Evaluation and interpretation

- Review ROC-AUC and average precision, calibration and Brier score, training cost and inference cost.
- Select candidates using validation performance rather than repeatedly choosing based on test results.
- For the MVP, use one prespecified operating objective (for example, target 80% signal efficiency) and choose the threshold on validation labels/scores. Report background rejection on test with that threshold unchanged.
- Inspect permutation importance for one strong classical model only if runtime permits.

**Exit gate:** tables/plots can be regenerated from saved outputs.

### Day 6 — Report and demo

- Write problem, dataset, methods, split, results, limitations and conclusion.
- Build a short demo: problem → dataset → experiment → metrics → lesson → limitation.
- Verify that every reported number appears in `results.csv`/metadata or can be recalculated.

**Exit gate:** all output files match the report.

### Day 7 — Buffer and viva practice

- Re-run all notebook cells from a clean runtime.
- Check links, filenames, output ZIP and app artifacts.
- Practice explaining false positives, false negatives, overfitting, why the test set is held out, why the model cannot prove a discovery, and why Higgs-generated mass is not the source of most proton mass.

## The 2-hour emergency path

1. 0–10 min: open the notebook, read intro Markdown, run imports.
2. 10–25 min: load 250k official rows and run integrity checks.
3. 25–70 min: run the classical baselines.
4. 70–100 min: run the MLP.
5. 100–115 min: inspect metrics and plots.
6. 115–120 min: download outputs and record any unfinished work.

This is only a rush plan. If all models cannot finish in the window, do not fake scores or conceal the unfinished models; present the baseline that genuinely completed and state what remains.

## What to say in the mid-sem report

“We are constructing a reproducible simulated HIGGS event-classification benchmark. The first phase evaluates Logistic Regression, XGBoost and an MLP across low-level, physicist-derived and combined features on a documented subset. The aim is to measure the performance/compute trade-off and establish whether a model is suitable for further study. Results are preliminary and are not a real-data discovery claim.”

## What to defer until after mid-sem

- Full 11-million-row experiments.
- Multi-seed or bootstrap uncertainty studies.
- Additional hyperparameter search.
- Advanced feature attribution such as SHAP.
- Authentication, deployment, cloud storage, user accounts or live detector feeds.
- Any “discovery significance” claim.

## Mid-sem acceptance checklist

- [ ] Meets the dataset budget or clearly documents the blocker and actual smaller run.
- [ ] Contains low-level, high-level and all-feature comparisons.
- [ ] Includes Logistic Regression, XGBoost, MLP; fallback if used is identified.
- [ ] Train/validation/test logic is documented and seed is fixed.
- [ ] Preprocessing uses training data only.
- [ ] Metrics and plots are generated by code.
- [ ] Model selection is based on validation, not test.
- [ ] Scientific and compute limitations are explicit.
- [ ] Team can explain the notebook rather than merely display it.
