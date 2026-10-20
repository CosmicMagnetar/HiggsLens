import numpy as np
import pandas as pd
import pytest

from higgslens.artifacts import matching_models, validate_prediction_frame
from higgslens.core import FEATURE_GROUPS, evaluate_frozen_candidate, make_subset_split, validate_higgs_array


def valid_raw(rows=20):
    raw = np.ones((rows, 29), dtype=float)
    raw[:, 0] = np.tile([0, 1], rows // 2 + 1)[:rows]
    return raw


def test_higgs_schema_and_feature_groups_are_exact():
    X, y = validate_higgs_array(valid_raw())
    assert X.shape == (20, 28)
    assert set(y) == {0, 1}
    assert len(FEATURE_GROUPS["low_level"]) == 21
    assert len(FEATURE_GROUPS["high_level"]) == 7
    assert len(FEATURE_GROUPS["all_features"]) == 28


@pytest.mark.parametrize(
    "raw", [np.ones((5, 28)), np.ones((0, 29)), np.column_stack([np.zeros(4), np.ones((4, 28))])]
)
def test_invalid_schema_or_missing_class_fails(raw):
    with pytest.raises(ValueError):
        validate_higgs_array(raw)


def test_nonfinite_and_nonbinary_values_fail():
    raw = valid_raw()
    raw[0, 3] = np.nan
    with pytest.raises(ValueError):
        validate_higgs_array(raw)
    raw = valid_raw()
    raw[0, 0] = 2
    with pytest.raises(ValueError):
        validate_higgs_array(raw)


def test_subset_split_is_deterministic_disjoint_and_exhaustive():
    y = np.tile([0, 1], 50)
    first, second = make_subset_split(y, 14), make_subset_split(y, 14)
    assert np.array_equal(first.train, second.train)
    assert len(set(first.train) & set(first.validation)) == 0
    assert len(set(first.train) | set(first.validation) | set(first.test)) == len(y)
    assert first.counts() == {"train": 64, "validation": 16, "test": 20}


def test_final_evaluation_preserves_validation_selected_threshold():
    validation_y = [0, 0, 0, 1, 1, 1]
    validation_p = [0.1, 0.3, 0.6, 0.4, 0.7, 0.9]
    test_y, test_p = [0, 0, 1, 1], [0.2, 0.8, 0.5, 0.95]
    result = evaluate_frozen_candidate(validation_y, validation_p, test_y, test_p, 0.8)
    assert result["threshold_selection"]["split"] == "validation"
    assert result["test_metrics"]["threshold"] == result["threshold_selection"]["threshold"]
    assert result["test_metrics"]["recall"] >= 0


def test_prediction_artifact_validation_and_model_match():
    valid = pd.DataFrame({"label": [0, 1], "model": [0.2, 0.8]})
    checked = validate_prediction_frame(valid)
    assert matching_models(checked, checked) == ["model"]
    with pytest.raises(ValueError):
        validate_prediction_frame(pd.DataFrame({"label": [0, 1], "model": [0.2, 1.1]}))
    with pytest.raises(ValueError):
        matching_models(checked, pd.DataFrame({"label": [0, 1], "other": [0.2, 0.8]}))
