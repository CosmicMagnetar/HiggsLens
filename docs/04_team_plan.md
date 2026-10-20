# Five-Person Team Plan

## Shared outcome

One project, one repo, one agreed evaluation protocol, one artefact schema. The team does not build five separate notebooks and merge them the night before submission.

| Person | Owns | Required outputs | Reviews |
|---|---|---|---|
| **Krishna** | Research design, experimental protocol, integration, product direction and final report | Research questions, split/threshold policy, final comparison, final narrative, demo, release checklist | Reviews all claims and merged interfaces |
| **Saumya Mishra** | Dataset and physics context, EDA, source/feature documentation | Dataset card, row/label/finite-value checks, class/feature plots, feature dictionary and source citations | Reviews that data statements and physics framing are accurate |
| **Aditya Rana** | Classical AML models | Logistic Regression and XGBoost runs across feature groups, result rows, timing and model settings | Reviews classical model fairness and baseline quality |
| **Rachit Gupta** | Deep-learning model | MLP runs, architecture/training config, training curves, predicted probabilities for validation/test | Reviews DL reproducibility and train-only scaling |
| **Ronit Singh** | Streamlit product and quality assurance | Dashboard, artifact loading, threshold interaction, error states, screenshot/demo path, tests | Tests end-to-end using another person's run files |

## Why Krishna should not do everything

Your highest-value role is to make the scientific decision logic consistent and ensure the work tells one defensible story. Delegate implementation; personally understand the split, metrics, result interpretation, and limitation decisions. In the viva, every team member should explain their own module and the whole project at a basic level.

## Artifact contract

Before any parallel implementation, agree to these names and columns:

- `results.csv`: one row per `(model, feature_set, seed, data_budget)`; include validation and test metrics, fit/inference time, chosen threshold and model configuration.
- `val_predictions.csv`: `event_id`, `label`, one named probability column per model or one row per `(event_id, model, probability)`; choose one schema and keep it stable.
- `test_predictions.csv`: same schema as validation predictions.
- `run_metadata.json`: data source, actual n, feature count/groups, split definition, seed, package versions, device, start time and elapsed time.
- `permutation_importance.csv`: `feature`, `importance_mean`, `importance_std`, `model`, `feature_set`, `evaluation_partition`.
- `summary.md`: generated factual summary with result table and caveats.

The current starter code uses its own defined prediction schema. If the team changes it, change the notebook and dashboard together and document the schema once in `docs/`.

## GitHub workflow

- `main`: should always be runnable and suitable for demo.
- Branches: `data/eda`, `models/classical`, `models/mlp`, `app/dashboard`, `docs/report`, `fix/<short-name>`.
- One meaningful change per pull request (PR).
- Every PR needs: purpose, files changed, test/run evidence, possible result change, and reviewer.
- No raw HIGGS dataset, generated large predictions, API keys or personal credentials in Git.
- At least one person other than the author reviews each PR.
- Do not rewrite shared history after collaborators have pulled; use PRs and ordinary merges.

## Milestones

### M0 — Agreement

Krishna writes the research question and split policy. Everyone reviews the exact dataset and feature-group definitions.

### M1 — Vertical slice

Saumya validates data; Aditya trains one Logistic Regression; Rachit trains one MLP; Ronit loads a tiny synthetic artifact into a skeleton dashboard; Krishna integrates one end-to-end result.

### M2 — Full mid-sem MVP

Classical model matrix, MLP comparison, EDA, metrics, plots, and initial report are completed. Do not work on advanced extensions until this gate passes.

### M3 — Robustness and final protocol

Add scale ladder/repeats and finalize operating-threshold protocol. Krishna freezes the method before final evaluation.

### M4 — Product hardening

Ronit validates inputs and exports. Saumya checks labels/feature semantics. Aditya and Rachit verify their models are recorded correctly. Krishna verifies interpretation and claims.

### M5 — Clean reproduction and release

A non-author follows README from scratch, then the team produces a tagged release and report.

## Meeting and review rhythm

- Short team sync twice a week: completed / next / blockers.
- Weekly review against acceptance gates.
- Review first the scientific contract and saved outputs, then UI appearance.
- Record decisions in `docs/decisions/` if they affect split, metric, data or architecture.

## Contribution evidence

Track work by PR/commit, artifact, review and report section. Avoid dividing work only by “pages” of the report because the main risk is whether the modules integrate. Every person should be able to show specific working evidence of their contribution.

## AI use policy

AI can help explain concepts, draft tests, refactor boilerplate, inspect errors and polish text. The human owner verifies code and numbers. Never allow the LLM to invent metric values, claim the MLP is best before a measured comparison, or cite a paper that has not been checked. Keep a brief note of substantial AI assistance if your course rules request it.
