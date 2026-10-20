"""Local-file benchmark runner for the required HiggsLens model matrix."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    FEATURE_GROUPS,
    evaluate_frozen_candidate,
    make_subset_split,
    validate_higgs_array,
)
from .models import (
    TrainedCandidate,
    feature_indices,
    train_compact_mlp,
    train_logistic_regression,
    train_xgboost,
)


def load_local_higgs(path: str | Path, rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load a local official HIGGS CSV/CSV.GZ without downloading or substitution."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"HIGGS file not found: {source}")
    frame = pd.read_csv(source, header=None, nrows=rows, compression="infer")
    return validate_higgs_array(frame.to_numpy())


def _predict(candidate: TrainedCandidate, features: np.ndarray) -> np.ndarray:
    columns = feature_indices(candidate.feature_group)
    if candidate.name.startswith("Compact MLP"):
        bundle = candidate.estimator
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is needed to evaluate the saved MLP bundle") from exc
        values = bundle["scaler"].transform(features[:, columns]).astype(np.float32)
        model = bundle["model"]
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(values).to(bundle["device"]))).cpu().numpy()
    return candidate.estimator.predict_proba(features[:, columns])[:, 1]


def run_subset_benchmark(
    raw_data: np.ndarray,
    output_dir: str | Path,
    seed: int = 42,
    include_xgboost: bool = True,
    include_mlp: bool = True,
) -> dict[str, Any]:
    """Run the fixed 3-model × 3-feature-group matrix on a validated subset.

    Selection happens entirely on validation ROC-AUC. Test predictions are only
    produced after the matrix is fixed and the validation winner is frozen.
    """
    X, y = validate_higgs_array(raw_data)
    split = make_subset_split(y, seed)
    train, validation, test = split.train, split.validation, split.test
    candidates: list[TrainedCandidate] = []
    for group in FEATURE_GROUPS:
        candidates.append(
            train_logistic_regression(X[train], y[train], X[validation], y[validation], group, seed)
        )
        if include_xgboost:
            candidates.append(train_xgboost(X[train], y[train], X[validation], y[validation], group, seed))
        if include_mlp:
            candidates.append(
                train_compact_mlp(X[train], y[train], X[validation], y[validation], group, seed)
            )
    selected = max(candidates, key=lambda item: item.details["validation"]["roc_auc"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation_predictions: dict[str, Any] = {"label": y[validation]}
    test_predictions: dict[str, Any] = {"label": y[test]}
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        validation_predictions[candidate.name] = candidate.validation_probabilities
        started = perf_counter()
        test_probability = _predict(candidate, X[test])
        inference_ms_per_1000 = (perf_counter() - started) * 1_000_000 / len(test)
        test_predictions[candidate.name] = test_probability
        results.append(
            {
                "Model": candidate.name,
                "Feature_set": candidate.feature_group,
                "Fit_seconds": candidate.fit_seconds,
                "Inference_ms_per_1000": inference_ms_per_1000,
                **{f"validation_{key}": value for key, value in candidate.details["validation"].items()},
            }
        )
    final = evaluate_frozen_candidate(
        y[validation], selected.validation_probabilities, y[test], test_predictions[selected.name]
    )
    results_frame = pd.DataFrame(results).sort_values("validation_roc_auc", ascending=False)
    results_frame.to_csv(output / "results.csv", index=False)
    pd.DataFrame(validation_predictions).to_csv(output / "val_predictions.csv", index=False)
    pd.DataFrame(test_predictions).to_csv(output / "test_predictions.csv", index=False)
    config = {"seed": seed, "feature_groups": list(FEATURE_GROUPS), "candidate_count": len(candidates)}
    metadata = {
        "run_kind": "subset_higgs_benchmark",
        "rows_loaded": len(X),
        "split": split.metadata(),
        "frozen_candidate": {"model": selected.name, "target_recall": 0.80},
        "final_evaluation": final,
        "candidate_details": {candidate.name: candidate.details for candidate in candidates},
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }
    (output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed HiggsLens subset benchmark from a local HIGGS CSV."
    )
    parser.add_argument("--data", required=True, help="Path to official HIGGS.csv or HIGGS.csv.gz")
    parser.add_argument("--output", required=True, help="New output directory for this run")
    parser.add_argument("--rows", type=int, default=250_000, help="Rows to load from the local source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--without-xgboost", action="store_true")
    parser.add_argument("--without-mlp", action="store_true")
    args = parser.parse_args()
    X, y = load_local_higgs(args.data, args.rows)
    raw = np.column_stack((y, X))
    run_subset_benchmark(raw, args.output, args.seed, not args.without_xgboost, not args.without_mlp)


if __name__ == "__main__":
    main()
