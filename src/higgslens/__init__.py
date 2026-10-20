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
]
