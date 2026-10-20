"""Strict artifact readers shared by the dashboard and smoke checks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_prediction_frame(frame: pd.DataFrame, name: str = "predictions") -> pd.DataFrame:
    """Validate a label plus probability columns without silently dropping rows."""
    if "label" not in frame.columns or len(frame) == 0:
        raise ValueError(f"{name} must be non-empty and contain a 'label' column")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{name} labels must be finite binary 0/1 values")
    models = [column for column in frame.columns if column != "label"]
    if not models:
        raise ValueError(f"{name} must contain at least one model probability column")
    checked = frame.copy()
    checked["label"] = labels.astype(int)
    for model in models:
        values = pd.to_numeric(checked[model], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{name} model '{model}' contains invalid probabilities")
        checked[model] = values.astype(float)
    return checked


def matching_models(validation: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Return shared model columns or fail clearly on a model-schema mismatch."""
    validation_models, test_models = set(validation.columns) - {"label"}, set(test.columns) - {"label"}
    if not validation_models & test_models:
        raise ValueError("validation and test prediction files have no matching model columns")
    if validation_models != test_models:
        raise ValueError(
            f"prediction model columns differ; missing from test={sorted(validation_models - test_models)}, missing from validation={sorted(test_models - validation_models)}"
        )
    return sorted(validation_models)
