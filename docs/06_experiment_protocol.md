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

For a subset MVP, use the fixed named `subset_stratified_64_16_20` policy: a stratified 64%/16%/20% split with seed recorded (default 42). For the stronger end-sem study, keep the same named policy across candidates, with a sufficiently large held-out test set. Record whether the development pool is a contiguous official-file prefix or a seeded sample; do not call it the official protocol.

1. Split labels and features into train/validation/test partitions.
2. Fit scaling on training rows only.
3. Apply the fitted transform to validation and test.
4. Train model parameters on training data.
5. Compare/tune model configurations using validation data.
6. Choose the operating threshold using validation data only.
7. Evaluate the frozen chosen configuration and threshold on test data.

For the optional `full_official_last_500k` protocol, preserve source-order rows 10,500,001–11,000,000 as the final test set and split only the preceding development rows. This needs the complete source dataset and extra disk/runtime; label the run accordingly. Do not claim to have used this official split for a simple random split from the first 250k rows.

## Model matrix

| Model | Feature groups | Preprocessing |
|---|---|---|
| Logistic Regression | Low-level, high-level, all | StandardScaler fitted on training only |
| XGBoost | Low-level, high-level, all | No standardization normally required for trees; use same rows/features and fixed settings |
| MLP | Low-level, high-level, all | StandardScaler fitted on training only; batch normalization + dropout |

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

Every run should save (under one run identifier):

- `results.csv`
- `val_predictions.csv`
- `test_predictions.csv`
- `run_metadata.json`
- `experiment_config.json` with frozen-candidate settings before final scoring
- `summary.md`
- ROC, precision–recall, calibration and feature-importance plots if generated

Use a separate output folder per run, e.g., `outputs/midsem_250k_seed42/` and `outputs/endsem_1m_seed42/`.

The dashboard requires `run_metadata.json` to declare `frozen_candidate.model` and `frozen_candidate.target_recall` before it will reveal final-test metrics. Its development controls use validation data only.
