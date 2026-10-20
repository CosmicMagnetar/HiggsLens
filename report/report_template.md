# HiggsLens: Particle-Event Classification via Expert Features and Neural Networks

## Abstract

Write this last. Include the task, dataset size/source, the compared models, the measured best result, the operating-point finding and the main limitation. Use only actual output values.

## 1. Introduction and motivation

Explain the Higgs field/boson at the level relevant to the task, the distinction between signal and background, and why ranking signal-like events matters. Do not claim that this project discovers the Higgs boson.

## 2. Problem statement and research questions

- Primary: classical AML versus MLP under a fixed data/compute budget.
- Feature ablation: low-level (21), high-level (7), all (28).
- Decision-oriented extension: background rejection at a validation-selected target signal efficiency.

## 3. Dataset

Record official UCI citation, simulated-data nature, actual number of events, label mapping, feature count, subset protocol, random seed and class distribution.

## 4. Methods

Describe the split, training-only preprocessing, Logistic Regression, XGBoost, MLP architecture, training settings, software/hardware and timing method.

## 5. Metrics and evaluation protocol

Define ROC-AUC, average precision, Brier score/calibration, accuracy (secondary), threshold metrics, fit/inference time and the validation-only threshold rule.

## 6. Results

Insert a table generated from `results.csv`. Include ROC and PR curves, calibration diagram, feature importance and one error-analysis figure if available.

## 7. Discussion

Answer the research questions. Separate observations from possible explanations. State if results differ by feature group or model family. Do not over-interpret feature importance as causal physics.

## 8. Product architecture and user workflow

Explain how a user loads run artifacts, compares candidates, inspects calibration, chooses an operating target and exports the report. Note that the app is a research/educational tool.

## 9. Limitations

Include simulated Monte Carlo data, sample size, available compute, absence of event-weight/yield/systematic treatment where applicable, class-prior mismatch, model generalization limits, and anything unfinished.

## 10. Conclusion and future work

Summarize measured outcomes; propose only next steps justified by the evidence.

## References

1. UCI Machine Learning Repository, HIGGS dataset: https://archive.ics.uci.edu/dataset/280/higgs
2. P. Baldi, P. Sadowski, D. Whiteson, “Searching for Exotic Particles in High-Energy Physics with Deep Learning,” *Nature Communications* 5, 4308 (2014). https://doi.org/10.1038/ncomms5308
3. CERN, “The Higgs boson.” https://home.cern/science/physics/higgs-boson/
4. Additional references only after you verify that you actually used them.
