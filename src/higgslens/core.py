"""Leakage-resistant primitives for reproducible HIGGS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

FEATURE_NAMES = (
    "lepton_pT",
    "lepton_eta",
    "lepton_phi",
    "missing_energy_magnitude",
    "missing_energy_phi",
    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet1_btag",
    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "jet2_btag",
    "jet3_pt",
    "jet3_eta",
    "jet3_phi",
    "jet3_btag",
    "jet4_pt",
    "jet4_eta",
    "jet4_phi",
    "jet4_btag",
    "m_jj",
    "m_jjj",
    "m_lv",
    "m_jlv",
    "m_bb",
    "m_wbb",
    "m_wwbb",
)
FEATURE_GROUPS = {
    "low_level": tuple(FEATURE_NAMES[:21]),
    "high_level": tuple(FEATURE_NAMES[21:]),
    "all_features": FEATURE_NAMES,
}


@dataclass(frozen=True)
class SplitIndices:
    """Index-only split contract, making partition overlap auditable."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int
    protocol: str = "subset_stratified_64_16_20"

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "validation": len(self.validation), "test": len(self.test)}

    def metadata(self) -> dict[str, Any]:
        return {**self.counts(), "seed": self.seed, "protocol": self.protocol}


def _as_binary_labels(y_true: Sequence[int]) -> np.ndarray:
    y = np.asarray(y_true).reshape(-1)
    if y.size == 0:
        raise ValueError("labels must be non-empty")
    try:
        numeric = y.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must contain numeric 0 and 1 values") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0, 1]).all():
        raise ValueError("labels must contain only finite 0 and 1 values")
    return numeric.astype(int)


def _as_probabilities(probabilities: Sequence[float], expected_size: int) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if p.size != expected_size or p.size == 0:
        raise ValueError("labels and probabilities must be non-empty arrays of equal length")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    return p


def validate_higgs_array(data: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Validate UCI's 29-column schema and return ``(features, labels)``."""
    array = np.asarray(data)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("dataset must be a non-empty two-dimensional array")
    if array.shape[1] != 29:
        raise ValueError("HIGGS input must have exactly 29 columns: label plus 28 features")
    try:
        array = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("HIGGS input must contain numeric values") from exc
    if not np.isfinite(array).all():
        raise ValueError("HIGGS input contains non-finite values")
    y = _as_binary_labels(array[:, 0])
    if np.unique(y).size != 2:
        raise ValueError("HIGGS input must contain both signal (1) and background (0)")
    return array[:, 1:], y


def make_subset_split(y_true: Sequence[int], seed: int = 42) -> SplitIndices:
    """Create a deterministic stratified 64/16/20 subset split with no overlap."""
    y = _as_binary_labels(y_true)
    if np.unique(y).size != 2:
        raise ValueError("both classes are required for a stratified split")
    indices = np.arange(len(y))
    development, test = train_test_split(indices, test_size=0.20, stratify=y, random_state=seed)
    train, validation = train_test_split(
        development, test_size=0.20, stratify=y[development], random_state=seed
    )
    split = SplitIndices(np.sort(train), np.sort(validation), np.sort(test), seed)
    validate_partition_indices(split, len(y))
    return split


def validate_partition_indices(split: SplitIndices, n_rows: int) -> None:
    parts = [split.train, split.validation, split.test]
    joined = np.concatenate(parts)
    if any(part.ndim != 1 or len(part) == 0 for part in parts):
        raise ValueError("each partition must be a non-empty one-dimensional index array")
    if (joined < 0).any() or (joined >= n_rows).any():
        raise ValueError("partition contains an out-of-range row index")
    if len(np.unique(joined)) != n_rows or len(joined) != n_rows:
        raise ValueError("partitions must be disjoint and collectively exhaustive")


def select_threshold_for_recall(
    y_true: Sequence[int], probabilities: Sequence[float], target_recall: float = 0.80
) -> float:
    """Choose the lowest-FPR threshold meeting recall, using validation data only."""
    y = _as_binary_labels(y_true)
    p = _as_probabilities(probabilities, len(y))
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    if np.unique(y).size != 2:
        raise ValueError("both classes are required for threshold selection")
    fpr, tpr, thresholds = roc_curve(y, p)
    eligible = np.flatnonzero(tpr >= target_recall)
    return float(thresholds[eligible[np.argmin(fpr[eligible])]]) if len(eligible) else float(np.min(p))


def summarize_at_threshold(
    y_true: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, float | int]:
    """Evaluate a *previously selected* threshold without modifying it."""
    y = _as_binary_labels(y_true)
    p = _as_probabilities(probabilities, len(y))
    if not np.isfinite(threshold) or not 0 <= float(threshold) <= 1:
        raise ValueError("threshold must be a finite value in [0, 1]")
    pred = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "signal_efficiency": float(recall_score(y, pred, zero_division=0)),
        "background_rejection": float(tn / max(tn + fp, 1)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "threshold": float(threshold),
    }


def ranking_metrics(y_true: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    """Metrics valid for a fixed candidate; does not perform model selection."""
    y = _as_binary_labels(y_true)
    p = _as_probabilities(probabilities, len(y))
    if np.unique(y).size != 2:
        raise ValueError("both classes are required for ranking metrics")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "accuracy_at_0_5": float(accuracy_score(y, p >= 0.5)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "prevalence": float(np.mean(y)),
    }


def evaluate_frozen_candidate(
    validation_labels: Sequence[int],
    validation_probabilities: Sequence[float],
    test_labels: Sequence[int],
    test_probabilities: Sequence[float],
    target_recall: float = 0.80,
) -> dict[str, Any]:
    """Freeze a validation-selected threshold, then apply it unchanged to test data."""
    threshold = select_threshold_for_recall(validation_labels, validation_probabilities, target_recall)
    return {
        "threshold_selection": {
            "split": "validation",
            "target_recall": target_recall,
            "threshold": threshold,
        },
        "validation_metrics": ranking_metrics(validation_labels, validation_probabilities),
        "test_metrics": {
            **ranking_metrics(test_labels, test_probabilities),
            **summarize_at_threshold(test_labels, test_probabilities, threshold),
        },
    }


def bootstrap_metric_interval(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    metric: str = "roc_auc",
    repetitions: int = 200,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Stratified bootstrap percentile interval for a final, frozen prediction set."""
    if repetitions < 20:
        raise ValueError("repetitions must be at least 20")
    y = _as_binary_labels(y_true)
    p = _as_probabilities(probabilities, len(y))
    functions = {"roc_auc": roc_auc_score, "average_precision": average_precision_score}
    if metric not in functions:
        raise ValueError("metric must be 'roc_auc' or 'average_precision'")
    if np.unique(y).size != 2:
        raise ValueError("both classes are required for bootstrap")
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = []
    for _ in range(repetitions):
        sampled = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        values.append(functions[metric](y[sampled], p[sampled]))
    return {
        "metric": metric,
        "repetitions": repetitions,
        "seed": seed,
        "estimate": float(functions[metric](y, p)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }
