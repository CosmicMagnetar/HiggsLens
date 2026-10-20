"""Small CPU-only end-to-end smoke run; never represents a HIGGS benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .core import FEATURE_NAMES, evaluate_frozen_candidate, make_subset_split, validate_higgs_array


def run(output_dir: str | Path = "outputs/smoke") -> dict:
    """Train one synthetic logistic model and write schema-compatible artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features, labels = make_classification(
        n_samples=400, n_features=28, n_informative=12, n_redundant=4, random_state=7
    )
    X, y = validate_higgs_array(np.column_stack([labels, features]))
    split = make_subset_split(y, seed=7)
    model = Pipeline(
        [("scaler", StandardScaler()), ("logistic", LogisticRegression(max_iter=300, random_state=7))]
    )
    model.fit(X[split.train], y[split.train])
    val_p, test_p = model.predict_proba(X[split.validation])[:, 1], model.predict_proba(X[split.test])[:, 1]
    final = evaluate_frozen_candidate(y[split.validation], val_p, y[split.test], test_p, target_recall=0.80)
    name = "Logistic Regression | all_features | synthetic smoke"
    pd.DataFrame({"label": y[split.validation], name: val_p}).to_csv(
        output / "val_predictions.csv", index=False
    )
    pd.DataFrame({"label": y[split.test], name: test_p}).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "Model": name,
                "Feature_set": "all_features",
                **final["validation_metrics"],
                **{f"test_{k}": v for k, v in final["test_metrics"].items()},
            }
        ]
    ).to_csv(output / "results.csv", index=False)
    metadata = {
        "run_kind": "synthetic_smoke_not_higgs_benchmark",
        "rows_loaded": len(X),
        "feature_count": len(FEATURE_NAMES),
        "split": split.metadata(),
        "frozen_candidate": {"model": name, "target_recall": 0.80},
        "final_evaluation": final,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    run()
