# Model training notes

HiggsLens uses the same train, validation, and test rows for all model families in a run. The data split is created before any scaler is fitted. Validation scores drive model selection and threshold choice; the test partition is evaluated only after that choice is recorded.

## Models included

- Logistic Regression is the scaled linear baseline.
- XGBoost uses histogram trees and validation-based early stopping.
- The compact MLP has two 128-unit layers, batch normalization, ReLU activations, dropout, weight decay, gradient clipping, and early stopping.

The MLP runner compares AdamW and AdaGrad over the same bounded setup. It keeps the optimizer with the strongest validation ROC-AUC for that feature group, records both trials in the run metadata, and restores the corresponding checkpoint. This is a model-development comparison, not a test-set competition.

Every model is run on the 21 low-level inputs, the seven engineered high-level inputs, and all 28 inputs. Results should be judged using validation and frozen-test ROC-AUC, average precision, calibration, operating-point metrics, uncertainty, and runtime together. A higher single score is not enough to claim that a model is generally better.

## Running a full matrix

Install the app dependencies, place the official UCI file locally, and run:

```bash
python -m higgslens.benchmark \
  --data data/raw/HIGGS.csv.gz \
  --output outputs/subset_250k_seed42 \
  --rows 250000 \
  --mlp-optimizers adamw,adagrad
```

The default run requires XGBoost and PyTorch. For a quick installation check only, use `--without-xgboost --without-mlp`; it is not an assignment benchmark because the required model matrix is incomplete.
