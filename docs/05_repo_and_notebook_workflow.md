# Repository, Folder Structure, and Notebook Workflow

## Recommended GitHub repository

Try: **`higgslens-higgs-event-classification`**. The shorter `higgslens` may already be taken; check availability before creating it. The repository name is a recommendation, not a claim that the name is available.

**Description:** Reproducible ML benchmark and exploratory dashboard for simulated Higgs signal/background classification.  
**Topics:** `higgs-boson`, `particle-physics`, `machine-learning`, `scikit-learn`, `xgboost`, `pytorch`, `streamlit`, `reproducible-research`.

Suggested visibility: private while the team is changing code, then public only if course policy permits. Put the dataset source/citation in README, but do not include the data file itself.

## Folder structure

```text
higgslens-higgs-event-classification/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── pyproject.toml
├── requirements-app.txt
├── requirements-dev.txt
├── data/
│   ├── raw/                 # downloaded HIGGS file; never commit
│   ├── processed/           # optional local cached subset; never commit
│   └── README.md            # how to obtain/verify data
├── notebooks/
│   ├── 01_midsem_mvp.ipynb
│   └── 02_endsem_benchmark.ipynb
├── src/
│   └── higgslens/
│       ├── __init__.py
│       └── core.py          # threshold selection and metric helpers
├── app/
│   └── higgs_lens_app.py
├── tests/
│   └── test_threshold_logic.py
├── docs/
│   ├── 00_START_HERE.md
│   ├── 01_higgs_physics_for_beginners.md
│   ├── 02_midsem_plan.md
│   ├── 03_endsem_excellence_plan.md
│   ├── 04_team_plan.md
│   ├── 05_repo_and_notebook_workflow.md
│   ├── 06_experiment_protocol.md
│   └── 07_demo_and_viva.md
├── reports/
│   ├── report_template.md
│   └── figures/              # only report-ready figures you choose to keep
└── outputs/                  # generated run artifacts; excluded from Git by default
```

## Why these folders exist

- `data/raw`: keeps large source data separate from code; it belongs in `.gitignore`.
- `data/processed`: local, derived tables if needed; record how they were generated.
- `notebooks`: learning, analysis and end-to-end experimental narrative. Markdown cells explain each operation.
- `src`: reusable logic that should be unit-tested and not copied across notebooks.
- `app`: the user-facing visualization layer; loads saved artifacts rather than silently retraining.
- `tests`: protects threshold, validation and utility logic.
- `docs`: physics background, protocol, plans and design decisions.
- `outputs`: metrics, predictions, metadata and figures produced by a run; keep big run artifacts out of the repo unless a small example is deliberately curated.

## Notebooks: text boxes instead of code comments

Every meaningful notebook section should use this order:

1. **Markdown heading** describing the stage.
2. **Markdown explanation box/section:** what this cell does, why it is needed, assumptions and the expected output.
3. **Code cell** containing only runnable code, with descriptive variable/function names and no explanatory `#` comments.
4. **Markdown interpretation note** explaining what the resulting table/plot does and does not establish.

Notebook explanations can use bold text, callouts, tables and equations. Avoid making code cells into long essays. Put reusable detailed explanations in `docs/` and link them from the notebook.

Example Markdown text cell:

> **Why this step matters**  
> We fit the scaler on training data only. If it learned its mean and standard deviation from the test set, information from the final evaluation would influence preprocessing and weaken the independence of the test result.

## Data policy

The official UCI HIGGS file is large. Use a documented subset first. Keep a local file such as `data/raw/HIGGS.csv.gz` out of Git. The small `.gitkeep` file only preserves an empty directory. Do not create a fake sample of results for demonstration; synthetic data may be used only for unit tests or an explicitly labelled UI test.

## Git commands for initial setup

Run these from the project folder after creating an empty repository on GitHub:

```bash
git init
git add README.md LICENSE CITATION.cff .gitignore docs notebooks src app tests requirements-app.txt requirements-dev.txt pyproject.toml
git commit -m "Initial HiggsLens project scaffold"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/higgslens-higgs-event-classification.git
git push -u origin main
```

Replace `<YOUR-USERNAME>` with your actual GitHub username and confirm the remote URL before pushing. Do not put the downloaded HIGGS data or the output folder in the `git add` command.

## Suggested commit sequence

- `docs: add physics overview and experiment protocol`
- `data: add HIGGS validation and subset loading`
- `models: add logistic regression and xgboost baselines`
- `models: add compact MLP baseline`
- `eval: add ROC PR calibration and timing report`
- `app: visualize benchmark artifacts and operating points`
- `test: add threshold selection tests`
- `docs: add mid-sem results and limitations`

## Local setup (from repository root)

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements-app.txt
pip install -r requirements-dev.txt
```

For Google Colab, upload the notebook and run cells in order. Colab's preinstalled libraries may differ; record the versions actually used. Restart a runtime only if the install cell instructs you to.

## Notebook versioning practices

- Keep large executed outputs out of Git when possible; clear them before committing if they make the notebook enormous.
- Commit the notebook structure and Markdown text cells so the teaching narrative remains.
- Save run results to `outputs/<run-id>/` and include run metadata.
- Use one seed for the first reproducing baseline; use additional documented seeds for the end-semester robustness study.
- Never hand-edit a results table after it is generated.
