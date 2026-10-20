"""Reusable, leakage-resistant utilities for HiggsLens experiments."""

from .core import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    SplitIndices,
    bootstrap_metric_interval,
    evaluate_frozen_candidate,
    make_subset_split,
    ranking_metrics,
    select_threshold_for_recall,
    summarize_at_threshold,
    validate_higgs_array,
)
from .models import (
    feature_indices,
    train_best_compact_mlp,
    train_compact_mlp,
    train_logistic_regression,
    train_xgboost,
)

__all__ = [
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
    "SplitIndices",
    "bootstrap_metric_interval",
    "evaluate_frozen_candidate",
    "make_subset_split",
    "ranking_metrics",
    "select_threshold_for_recall",
    "summarize_at_threshold",
    "validate_higgs_array",
    "feature_indices",
    "train_best_compact_mlp",
    "train_compact_mlp",
    "train_logistic_regression",
    "train_xgboost",
]
