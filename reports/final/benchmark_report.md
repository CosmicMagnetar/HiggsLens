# HiggsLens benchmark report

## Status

**[NOT RUN] Real HIGGS benchmark.** No official dataset or prior output artifacts were present in the audited checkout, and the host did not have the project dependencies installed. This report intentionally contains no invented HIGGS metrics, runtime, hardware, or result table.

## Objective and protocol

HiggsLens compares Logistic Regression, XGBoost, and a compact PyTorch MLP over low-level (21), high-level (7), and all (28) UCI HIGGS features. Candidate selection is validation-only; final results require a metadata-declared frozen candidate and threshold. See [experimental design](../../docs/research/experimental_design.md) and [audit](../audit/baseline_and_code_audit.md).

## Required evidence before replacing this status

- Dataset provenance, checksum where practical, exact row count and partition boundaries.
- Saved `experiment_config.json`, `run_metadata.json`, validation and final-test predictions, `results.csv`, timing definition/device, and recorded configurations.
- Validation selection table; final ROC-AUC, AP, Brier, log loss, prevalence, frozen operating-point metrics, and bootstrap intervals traceable to the saved predictions.
- Calibration/ROC/PR figures and any validation-only permutation importance.

## Limitations

The HIGGS data are simulated Monte Carlo events. Scores are benchmark quantities, not discovery evidence, real-event probabilities, detector deployment performance, or physical causation claims.
