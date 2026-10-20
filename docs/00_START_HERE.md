# HiggsLens: Start Here

**Project owner:** Krishna  
**Team:** Krishna, Saumya Mishra, Aditya Rana, Rachit Gupta, Ronit Singh  
**Proposed repository:** `higgslens-higgs-event-classification`  
**Project type:** Applied machine learning + scientific benchmark + educational web tool

## The one-sentence idea

HiggsLens compares classical machine-learning models and a compact neural network on simulated HIGGS collision events, then helps a user inspect the trade-off between retaining signal-like events, rejecting background-like events, and compute cost.

## What this project does and does not claim

It **does** provide a reproducible classification benchmark, a fair comparison of 21 low-level and 7 physicist-derived features, measured evaluation results, and an exploratory dashboard. It **does not** simulate the Higgs field, prove a new discovery, or qualify as a CERN production analysis. The provided HIGGS dataset is simulated Monte Carlo data. A real discovery analysis needs additional detector/physics context, event weights, expected yields, systematic uncertainties, and a formal statistical model.

## What the supplied assignment requires

The attached brief's Project 10 specifies:

- Task: classify simulated collision events into signal and background.
- Classical AML baseline: Logistic Regression and XGBoost, evaluated on low-level and expert-derived feature groups.
- DL model: compact multilayer perceptron (MLP) with batch normalization and dropout.
- Fair comparison: fixed data budget and a held-out test set; compare ROC-AUC, PR-AUC, calibration, and training time.
- Dataset: UCI HIGGS; begin with a documented stratified subset of 250,000 to 1,000,000 events before attempting all 11 million.
- Implementation advice: read in chunks, separate the label before scaling, fit preprocessing only on training data, set an explicit random seed, and compare models at the same data budget.
- Expected deliverable: scalable benchmark showing accuracy, calibration, and compute trade-offs.

These requirements are the academic core. Dashboard features and extra robustness tests are additions, not replacements.

## Choose the correct version

| Version | Data budget | Goal | What to submit |
|---|---:|---|---|
| Mid-sem MVP | 250,000 events | Demonstrate the complete methodology and the baseline comparison quickly | Executed notebook, EDA, baseline table, MLP result, plots, initial report, short demo |
| End-sem excellence | 1,000,000 events as the target, scale down transparently if compute fails | Improve reproducibility, robustness, threshold analysis, uncertainty, and product usability | Frozen benchmark, ablations, test report, dashboard, final paper/report, demo, test evidence |
| Optional stretch | Full data / official final-500,000 test protocol if storage and runtime allow | Stronger protocol matching the original benchmark | Separate run metadata and explicit statement of data split |

The end-sem version need not run on every row to be excellent. It must be honest, methodologically fair, reproducible, and clearly document the exact event count and split. The assignment itself recommends beginning with a subset.

## First 30 minutes

1. Create a GitHub repository with the suggested name (check whether it is available before creation).
2. Download and extract this project pack.
3. Open `notebooks/01_midsem_mvp.ipynb` in Google Colab.
4. Read the introductory Markdown cells before running code.
5. Run the notebook top-to-bottom, checking that the official UCI source and actual row count are printed.
6. Save `outputs/` artifacts, but never commit the raw HIGGS data or large model files.

## Suggested GitHub setup

**Repository name:** `higgslens-higgs-event-classification`  
**Short description:** `Reproducible ML benchmark and exploratory dashboard for simulated Higgs signal/background classification.`  
**Topics:** `higgs-boson`, `particle-physics`, `machine-learning`, `scikit-learn`, `xgboost`, `pytorch`, `streamlit`, `reproducible-research`.

Use a public repository if your course allows it; otherwise make it private until grading/submission rules are clear. Do not create an empty public repository containing unverified scores or private course details. Check your institution's rules about AI assistance and public code.

## What excellence means here

Excellence is not “the MLP must win.” Excellence means a well-framed question, fair data splitting, no leakage, explicit baselines, suitable metrics, measured compute cost, controlled threshold choice, uncertainty/robustness checks, clear limitations, polished artefacts, and a product that someone else can run.

## Source-of-truth files

- `docs/01_higgs_physics_for_beginners.md`: physics background in beginner language.
- `docs/02_midsem_plan.md`: mid-semester execution plan.
- `docs/03_endsem_excellence_plan.md`: the full plan through end-semester.
- `docs/04_team_plan.md`: responsibilities for all five team members.
- `docs/05_repo_and_notebook_workflow.md`: folder structure, GitHub, notebooks, and AI workflow.
- `docs/06_experiment_protocol.md`: dataset, feature sets, splits, metrics, and analysis rules.
- `docs/07_demo_and_viva.md`: demo storyline and viva preparation.
- `docs/10_submission_checklists.md`: final mid-semester and end-semester evidence checks.
