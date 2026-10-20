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


<!-- PAGEBREAK -->

# Higgs Physics and the Machine-Learning Project — From Zero

## 1. What is the Higgs boson?

The Higgs boson is a particle associated with the Higgs field. In quantum field theory, fields are fundamental descriptions and particles can be understood as quantized excitations of fields. A useful first picture is that the Higgs field fills space, while a Higgs boson is an excitation of that field that can be produced in sufficiently energetic collisions. The field and the boson are related, but they are not identical.

The nickname **“God particle”** is popular media language, not a technical physics term. For your project report and viva, say “Higgs boson” or “Higgs field.” CERN's ATLAS and CMS experiments announced a new particle consistent with the Higgs boson on 4 July 2012.

## 2. Does the Higgs field give everything mass?

Not quite. The Brout–Englert–Higgs mechanism explains why the W and Z bosons are massive, and interactions with the Higgs field are associated with the masses of fundamental fermions such as electrons and quarks. Photons do not acquire mass through this mechanism. Also, most of the mass of a proton or neutron comes from the energy of the strong interaction (QCD) among quarks and gluons, not directly from the Higgs-generated rest masses of their constituent quarks.

So the accurate beginner-level summary is:

> The Higgs field is associated with the masses of many elementary particles through their interactions with it, but it does not explain every form of mass in ordinary matter.

Avoid saying that the Higgs boson itself “pushes mass into objects” or that detecting one simulated event proves that the Higgs field exists.

## 3. How was the Higgs boson observed?

The Higgs boson is extremely short-lived. Experiments do not see it as a little object sitting inside a detector. Instead, a high-energy collision may produce a Higgs boson, which quickly decays into other particles. Detectors measure the resulting tracks and energy deposits. Physicists reconstruct useful quantities from these measurements and search statistically for patterns that are more consistent with signal than with known background processes.

There are vastly more background processes and ordinary collisions than signal events. This is why efficient signal/background discrimination is a meaningful machine-learning task.

## 4. What the HIGGS dataset represents

The UCI HIGGS benchmark contains simulated Monte Carlo collision events for a binary classification task. It has 11,000,000 rows, 28 numerical input features, and a binary target label. The data are simulated, not raw detector readouts from a new discovery.

- `label = 1`: signal process that produces Higgs bosons.
- `label = 0`: background process that does not produce Higgs bosons.
- First 21 input features: low-level kinematic quantities associated with leptons, missing energy, and jets.
- Final 7 input features: physicist-derived high-level kinematic quantities intended to help discriminate signal from background.

A feature is a numerical input used by the model. The label is the answer the model is taught to predict. The exact feature names and grouping are in `docs/06_experiment_protocol.md` and in the notebook.

## 5. What question are we solving?

The core research question is:

> Under a fixed data and compute budget, how do classical models and a compact neural network compare when they use low-level features, expert-derived features, or both?

A second, product-oriented question is:

> Once a model gives each event a score, how does the trade-off between signal efficiency and background rejection change as we change the operating threshold?

The first question matches the assignment. The second turns the results into a useful exploratory workbench.

## 6. Why include different models?

### Logistic Regression

A useful transparent baseline. It learns a weighted combination of input features and maps it to a probability-like score. Because it is a relatively simple model, it helps show whether a more complex method is needed.

### XGBoost

A gradient-boosted decision-tree model. It can learn nonlinear relationships and interactions in tabular data. It is a strong classical AML comparator for the project.

### MLP neural network

A multilayer perceptron learns combinations of numerical inputs through trainable layers. Batch normalization can help stabilize activations during training, and dropout regularizes the network by randomly dropping activations during training. This is the DL model specified by the brief.

We compare all three on the same data split. Do not assume that deep learning must win: if XGBoost wins on the measured dataset, that is a valid scientific result.

## 7. Why compare low-level and high-level features?

The seven expert-derived variables were constructed from the lower-level measurements using physics knowledge. Comparing the groups tests whether that feature engineering benefits these model families. The fair comparisons are:

1. Low-level only (21 features).
2. High-level only (7 features).
3. All features (28 features).

Keep the feature groups identical across the eligible models wherever the project plan says they should be compared. Do not quietly remove difficult features from one model only.

## 8. Essential machine-learning vocabulary

- **Training set:** data used to fit model parameters.
- **Validation set:** data used to compare candidates, tune model settings, and choose an operating threshold.
- **Test set:** held back until the final assessment. It must not influence feature selection, hyperparameters, or threshold choice.
- **Data leakage:** information from validation/test data improperly influencing training or decisions; it makes results look better than they really are.
- **ROC-AUC:** evaluates ranking across many classification thresholds; higher is generally better.
- **Average precision / PR-AUC summary:** summarizes precision–recall performance; it is useful alongside ROC-AUC.
- **Calibration:** whether scores interpreted as probabilities agree with observed positive rates. Brier score is one measure; lower is better for the same evaluation data.
- **Signal efficiency / recall:** fraction of signal events retained at a chosen threshold.
- **Background rejection / specificity:** fraction of background events rejected at that threshold.
- **Training time:** elapsed model-fit time under the recorded runtime environment.

## 9. What your model can claim

It may claim: “On the documented simulated HIGGS subset and held-out split, model A achieved the measured ROC-AUC/AP/Brier score and a measured background rejection at the predeclared signal-efficiency target.”

It may not claim: “We discovered the Higgs boson,” “this model is ready for CERN,” or “this score gives the actual probability that a real collision contains a Higgs” without an appropriate real-data and physics-statistical analysis.

## Sources for the physics explanation

- CERN, *The Higgs boson*: https://home.cern/science/physics/higgs-boson/
- CERN, *What’s so special about the Higgs boson?*: https://home.cern/science/physics/higgs-boson/what/
- CERN announcement, 4 July 2012: https://home.cern/cern-experiments-observe-particle-consistent-with-long-sought-higgs-boson/


<!-- PAGEBREAK -->

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


<!-- PAGEBREAK -->

# Dataset and Experiment Protocol

## Dataset provenance and target

**Dataset:** UCI HIGGS  
**Official source:** https://archive.ics.uci.edu/dataset/280/higgs  
**Original paper:** P. Baldi, P. Sadowski, D. Whiteson, “Searching for Exotic Particles in High-Energy Physics with Deep Learning,” *Nature Communications* 5, 4308 (2014), https://doi.org/10.1038/ncomms5308  
**Data type:** simulated Monte Carlo event table  
**Original size:** 11,000,000 events  
**Inputs:** 28 continuous/numeric features  
**Target:** label 1 = signal; label 0 = background  
**Starter budget from assignment:** stratified subset of 250,000 to 1,000,000 events; record exact sample and seed.

## Input feature groups

Low-level features are columns 1–21 after the label in the original column order:

| Group | Feature names | Count |
|---|---|---:|
| Lepton | `lepton_pT`, `lepton_eta`, `lepton_phi` | 3 |
| Missing energy | `missing_energy_magnitude`, `missing_energy_phi` | 2 |
| Jets 1–4 | Each jet has `pt`, `eta`, `phi`, `btag` (`jet1_pt` ... `jet4_btag`) | 16 |
| High-level expert-derived | `m_jj`, `m_jjj`, `m_lv`, `m_jlv`, `m_bb`, `m_wbb`, `m_wwbb` | 7 |
| Total | Low-level + high-level | 28 |

The dataset fields are numeric representations, not raw detector images. `pt` denotes transverse momentum, `eta` pseudorapidity, `phi` azimuthal angle, and `m_*` labels derived kinematic quantities; use the official variable descriptions for detailed physics definitions.

## Dataset checks before model training

- Raw row has 29 values including label.
- Exactly 28 features after removing the label.
- Binary labels contain both classes and the mapping is recorded.
- All feature values are finite. If a check fails, stop and inspect rather than silently imputing.
- Source, retrieval method, actual row count, seed, split, package versions and device are written to run metadata.
- A sample below 250k may be used for a smoke test only; it must not be passed off as the planned benchmark.

## Splitting and leakage prevention

For a subset MVP, use a fixed stratified train/validation/test split such as 64%/16%/20%. For the stronger end-sem study, keep the same named policy across candidates, with a sufficiently large held-out test set. The exact split is less important than fixing it before model comparison and documenting it.

1. Split labels and features into train/validation/test partitions.
2. Fit scaling on training rows only.
3. Apply the fitted transform to validation and test.
4. Train model parameters on training data.
5. Compare/tune model configurations using validation data.
6. Choose the operating threshold using validation data only.
7. Evaluate the frozen chosen configuration and threshold on test data.

If the team later implements the original published HIGGS protocol, preserve the original final 500,000 rows as a separate test set. This needs the complete source dataset and extra disk/runtime; label the run accordingly. Do not claim to have used this official split for a simple random split from the first 250k rows.

## Model matrix

| Model | Feature groups | Preprocessing |
|---|---|---|
| Logistic Regression | Low-level, high-level, all | StandardScaler fitted on training only |
| XGBoost | Low-level, high-level, all | No standardization normally required for trees; use same rows/features and fixed settings |
| MLP | Low-level and all as a minimum; high-level optional if budget permits | StandardScaler fitted on training only; batch normalization + dropout |

For the end-sem report, explain any omitted feature-set/model combination and why. Do not silently substitute another tree model: if XGBoost cannot be used and a fallback is necessary, label the fallback in all tables and figures.

## Metrics

| Metric | Role | Interpretation |
|---|---|---|
| ROC-AUC | Primary ranking measure | Higher means better positive-over-negative ranking across thresholds |
| Average Precision | PR summary | Useful complement to ROC-AUC; depends on positive prevalence |
| Brier score | Calibration/probability error | Lower is better on the same target/data distribution |
| Reliability diagram | Calibration visualization | Compares average predicted probability with observed positive fraction |
| Accuracy | Secondary context | Can hide trade-offs; never report alone |
| Signal efficiency/recall | Operating point | Fraction of signal events retained at chosen threshold |
| Background rejection/specificity | Operating point | Fraction of background events rejected |
| Precision, F1, confusion matrix | Operating point | Explain selected-threshold error trade-offs |
| Fit/prediction time | Cost | Must name hardware, data size, software version and timing definition |

## Operating-point procedure

Declare a target signal efficiency (for example, 80%) before viewing test results. On validation data, select the threshold that meets the target while minimizing false-positive rate. Freeze that threshold. Apply it to test predictions without adjustment, then report test signal efficiency and background rejection. Small differences between validation and test performance are expected.

This selection metric is a useful student-facing operating objective, not the full objective of a particle-discovery analysis. Real sensitivity/significance estimates require inputs not supplied by the simple feature table, including appropriately normalized event weights, expected yields and systematic uncertainties.

## Calibration notes

The classifier output is a score/probability estimate under the simulated benchmark distribution. It is not automatically the real-world probability that a real detector event is a Higgs event. The simulated signal/background class prior may not match a real experimental stream. Show calibration on held-out benchmark data and clearly state that boundary.

## Explainability

Start with model coefficients (for Logistic Regression), permutation importance on a validation subset, and error analysis. SHAP is optional and should not displace the required experiment. Feature importance indicates predictive contribution under a method and dataset; it does not prove a physical causal relationship.

## Artifact contract

Every run should save:

- `results.csv`
- `val_predictions.csv`
- `test_predictions.csv`
- `run_metadata.json`
- `summary.md`
- ROC, precision–recall, calibration and feature-importance plots if generated

Use a separate output folder per run, e.g., `outputs/midsem_250k_seed42/` and `outputs/endsem_1m_seed42/`.


<!-- PAGEBREAK -->

# Demo Script and Viva Preparation

## Five-minute demo storyline

### 0:00–0:40 — Motivation

“The Higgs field is associated with the masses of many elementary particles. In high-energy collision experiments, a Higgs signal can be difficult to separate from background. This project studies that classification problem on simulated HIGGS data; it does not claim a new discovery.”

### 0:40–1:20 — Data

Show the official UCI citation, actual row count, 28 features, target labels, class balance and split. Explain the difference between 21 low-level and seven high-level expert-derived variables.

### 1:20–2:20 — Models

Show Logistic Regression, XGBoost and the MLP configuration. Explain that all candidates use identical partitions and the intended feature groups. Point out that tree models do not normally need standardization while the linear model and MLP do.

### 2:20–3:20 — Results

Show the result table, ROC/PR curves and calibration plot. State the measured top result using the values in the saved output. Also name the cost trade-off and any metric where the “winner” is not best.

### 3:20–4:20 — Product decision

Move the target signal-efficiency control. Explain that the operating threshold is selected from validation scores, then tested unchanged. Show how background rejection and the confusion matrix change. Do not adjust the threshold using test labels.

### 4:20–5:00 — Limitations

Explain that the benchmark is simulated, the data budget is limited relative to the full dataset if applicable, and the simple dataset lacks information required for full physical discovery significance. Close with next steps supported by your actual findings.

## Likely viva questions and model answers

**Q1. What is the difference between the Higgs field and the Higgs boson?**  
The Higgs field is a field present throughout space and is associated with masses of certain elementary particles through interactions. The Higgs boson is a quantum excitation of that field.

**Q2. Is the Higgs boson called the God particle in scientific literature?**  
That is a popular nickname, not a formal physics term. I use “Higgs boson” in the report.

**Q3. Does the Higgs field explain all the mass of your body?**  
No. It explains mass for elementary particles through the Higgs mechanism, but most proton/neutron mass comes from strong-interaction energy.

**Q4. What does the model classify?**  
It predicts the label of a simulated event as signal or background based on 28 numerical features.

**Q5. Is this a Higgs discovery?**  
No. It is a benchmark classification experiment on simulated data. It does not analyze actual detector data or compute a discovery significance.

**Q6. Why use Logistic Regression?**  
It is a transparent baseline that establishes how a relatively simple linear classifier performs before using more complex models.

**Q7. Why use XGBoost?**  
Boosted trees can capture nonlinear feature interactions and are a strong classical model for tabular data.

**Q8. Why use an MLP?**  
It is the compact neural model required by the assignment and allows a comparison between learned nonlinear combinations and the classical baselines.

**Q9. Why compare low-level and high-level features?**  
The last seven features encode physics-engineered kinematic information. We test whether this engineering helps models compared with using the low-level measurements alone.

**Q10. Why not use accuracy alone?**  
Accuracy depends on threshold and class balance. ROC-AUC and average precision summarize ranking, Brier/calibration assess probability quality, and threshold metrics describe the operational trade-off.

**Q11. What is data leakage in this project?**  
For example, fitting a scaler on all rows or choosing a threshold after looking at test labels lets final-evaluation information affect the pipeline and makes the test result less trustworthy.

**Q12. Why does the MLP use batch normalization and dropout?**  
Batch normalization helps stabilize intermediate activations during training; dropout is a regularization technique that reduces co-adaptation and can help control overfitting.

**Q13. What does an 80% signal-efficiency threshold mean?**  
A threshold is chosen on the validation set to retain about 80% of signal labels while minimizing false positives among thresholds meeting that target. We then apply that fixed threshold to the held-out test set and report what happens there.

**Q14. What is the difference between signal efficiency and precision?**  
Signal efficiency/recall is the fraction of true signal events that are retained. Precision is the fraction of events predicted as signal that are actually labeled signal in the evaluation data.

**Q15. What happens if Logistic Regression or XGBoost performs better than the MLP?**  
That is a valid result. We report it and discuss model performance, calibration, interpretability and compute cost without forcing the intended model to win.

**Q16. Why can't the app report the true real-world probability of a collision containing a Higgs?**  
The benchmark is simulated and has its own class prevalence. A real experimental stream has different rates and needs domain-specific calibration and physics analysis.

**Q17. How would you improve the project next?**  
Depending on the results, repeat key runs across seeds, scale the data budget, quantify metric uncertainty, independently reproduce a run, and study calibration/feature explanations. A real discovery analysis would require event weights, yields and systematics beyond the benchmark table.

## Demo checklist

- [ ] Clean browser/runtime and no hidden failing cells.
- [ ] Report numbers match the exported results.
- [ ] Dataset source and sample size visible.
- [ ] Threshold policy visible and described accurately.
- [ ] Simulated-data limitation visible.
- [ ] Backup screenshots/outputs available in case internet fails.
