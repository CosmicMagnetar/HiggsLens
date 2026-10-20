# HiggsLens experimental design

## Protocols

`subset_stratified_64_16_20` is for a fixed 250,000–1,000,000-event course subset. Record whether rows are a contiguous official-file prefix or sampled development pool, source URL/path, checksum where practical, seed 42 unless changed before the run, and exact counts. Split once, stratified: 64% train, 16% validation, 20% test.

`full_official_last_500k` is optional and requires all 11m official rows in source order: reserve rows 10,500,001–11,000,000 as untouched final test, then split the preceding development rows into train/validation. Do not shuffle across that boundary.

Validate 29 finite numeric columns before extracting label, 28 feature columns after, binary labels with both classes in every partition, and non-overlapping/exhaustive row indices. Persist source, schema/order, sampling policy, split metadata, package versions, device, runtime and Git SHA.

## Development → freeze → final evaluation

1. Develop only on training/validation. Fit scalers on train data; compare the predeclared model matrix on identical partition rows and feature groups.
2. Freeze one declared candidate from validation ROC-AUC, its hyperparameters/seed/feature group, calibration decision, and validation-selected threshold for target recall 0.80.
3. Produce final-test predictions once for fixed, prespecified candidates. Apply the frozen threshold unchanged. A later configuration change is a new exploratory experiment, not confirmation.

The dashboard accepts validation-only development artifacts. It exposes test metrics only when metadata names `frozen_candidate.model` and `frozen_candidate.target_recall`.

## Predeclared candidate budget

Run Logistic Regression (`StandardScaler` + `LogisticRegression`, solver/`C` recorded), XGBoost histogram candidates (maximum 12 combinations across depth 3/5/7, learning rate 0.03/0.1, subsample/colsample 0.8/1.0, `min_child_weight`, `reg_lambda`; validation early stopping), and a compact PyTorch MLP (two 128-unit layers, batch norm, ReLU, dropout, weight decay, patience and minimum delta recorded). Each must cover `low_level`, `high_level`, and `all_features`; omissions are [NOT RUN] rather than silently removed.

Record preprocessing, fit, and repeated inference timing separately. Report milliseconds per 1,000 events; synchronize GPU timing if used and state batch size/device. Bootstrap final ROC-AUC and AP with a declared stratified repetition count (200 default in the helper) and seed. Use validation permutation importance for one frozen classical candidate, on a bounded subset.
