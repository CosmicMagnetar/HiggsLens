"""Small, testable core helpers shared by the HiggsLens dashboard."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, recall_score, f1_score


def select_threshold_for_recall(y_true, probabilities, target_recall: float = 0.80) -> float:
    """Choose a threshold on a validation set that reaches target recall with minimum FPR.

    This function does not know whether inputs are validation or test data. The caller
    must pass validation labels/scores here and transfer the returned threshold unchanged
    to the test set. That rule is enforced by the pipeline / dashboard workflow.
    """
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.size == 0 or p.size != y.size:
        raise ValueError("y_true and probabilities must be non-empty arrays of equal length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    if np.unique(y).size != 2:
        raise ValueError("y_true must contain both classes for threshold selection")

    fpr, tpr, thresholds = roc_curve(y, p)
    eligible = np.flatnonzero(tpr >= target_recall)
    if not len(eligible):
        # ROC thresholds include a point that predicts every row positive; this fallback
        # is defensive for unusual versions/data edge cases.
        return float(np.min(p))
    best_idx = eligible[np.argmin(fpr[eligible])]
    return float(thresholds[best_idx])


def summarize_at_threshold(y_true, probabilities, threshold: float) -> dict:
    """Compute binary selection metrics at a frozen threshold."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.size == 0 or p.size != y.size:
        raise ValueError("y_true and probabilities must be non-empty arrays of equal length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    pred = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "signal_efficiency": float(recall_score(y, pred, zero_division=0)),
        "background_rejection": float(tn / max(tn + fp, 1)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
    }
