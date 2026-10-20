# HIGGS ML literature and benchmark review

## Source review

| Type | Source | Date | Relevant finding | Caveat and HiggsLens implication |
|---|---|---:|---|---|
| Official dataset | [UCI HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs), D. Whiteson | 2014 | 11,000,000 simulated events; label then 28 real-valued features; first 21 low-level and seven derived high-level; final 500,000 are the benchmark test examples. | The 2.6 GB compressed file is a distinct full-data protocol. A random course subset must not be called the official split. |
| Peer-reviewed paper | [Baldi, Sadowski & Whiteson, *Nature Communications* 5, 4308](https://doi.org/10.1038/ncomms5308) / [preprint](https://arxiv.org/abs/1402.4735) | 2014 | Studies deep learning against high-energy-physics classification methods on simulated exotic-particle processes. | It motivates low-/high-level ablations, not copying a score from a different split or implementation. |
| Official documentation | [scikit-learn: common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) | accessed 2026-10-21 | Split before learned preprocessing; fit transforms only on train data; pipelines reduce leakage risk. | StandardScaler is fit only on training rows; candidate and threshold choices use validation. |
| Official documentation | [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | accessed 2026-10-21 | Brier/log loss reflect calibration, discrimination, and uncertainty; reliability plots are needed for calibration inspection. | Report Brier with reliability diagrams, never label it pure calibration. Fit any calibrator without final-test labels. |
| Official documentation | [XGBoost sklearn interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html) | accessed 2026-10-21 | Manual evaluation split is required for early stopping; `tree_method="hist"` is supported; best iteration is used by prediction with early stopping. | Use validation as `eval_set`, bounded trial list, and record best iteration. Never use final test as `eval_set`. |

## Answers shaping the implementation

The defensible primary metric is ROC-AUC, complemented by AP (with prevalence), log loss, Brier score, calibration diagrams, and a prespecified operating point. Logistic Regression with training-only standardization is an interpretable baseline; histogram boosted trees and a regularized compact MLP are complementary numerical-tabular baselines. The required experiments remain Logistic Regression, XGBoost, and a PyTorch MLP across 21, 7, and 28 features.

For XGBoost, use a modest predeclared grid and validation ROC-AUC selection with `early_stopping_rounds`; persist every attempted configuration and selected `best_iteration`. For the MLP, fix seeds where practical, scale using train rows, use batch normalization, dropout and weight decay, monitor validation loss/AUC, and restore the best validation checkpoint. Repeat predeclared seeds when resources allow; report variation rather than treating tiny differences as a win.

The official full-data convention reserves the final 500,000 rows. Course subset mode instead uses a declared, stratified 64/16/20 split from a fixed loading policy. These modes are not interchangeable.

Bootstrap intervals on frozen final predictions quantify sampling uncertainty but do not license re-tuning. They should use a declared seed and repetition count. Feature importance is model reliance on the selected validation sample, not physics causality.
