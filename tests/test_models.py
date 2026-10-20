from sklearn.datasets import make_classification

from higgslens.core import make_subset_split
from higgslens.models import feature_indices, train_logistic_regression


def test_feature_groups_have_expected_indices():
    assert len(feature_indices("low_level")) == 21
    assert len(feature_indices("high_level")) == 7
    assert len(feature_indices("all_features")) == 28


def test_logistic_training_uses_a_feature_group_and_validation_only_metrics():
    X, y = make_classification(n_samples=120, n_features=28, n_informative=10, random_state=2)
    split = make_subset_split(y, seed=2)
    candidate = train_logistic_regression(
        X[split.train], y[split.train], X[split.validation], y[split.validation], "high_level", seed=2
    )
    assert candidate.validation_probabilities.shape == (len(split.validation),)
    assert 0 <= candidate.details["validation"]["roc_auc"] <= 1
    assert candidate.estimator.named_steps["scaler"].mean_.shape == (7,)
