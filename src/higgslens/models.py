"""Training implementations for the required HiggsLens model families.

Optional libraries are imported inside their respective functions so a basic
installation can still validate data and run the logistic-regression baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .core import FEATURE_GROUPS, FEATURE_NAMES, ranking_metrics


def feature_indices(feature_group: str) -> np.ndarray:
    """Return ordered column indices for a documented HIGGS feature group."""
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"unknown feature group {feature_group!r}; choose from {sorted(FEATURE_GROUPS)}")
    positions = {name: index for index, name in enumerate(FEATURE_NAMES)}
    return np.array([positions[name] for name in FEATURE_GROUPS[feature_group]], dtype=int)


@dataclass
class TrainedCandidate:
    """Fitted candidate plus development metadata used by the benchmark runner."""

    name: str
    feature_group: str
    estimator: Any
    validation_probabilities: np.ndarray
    fit_seconds: float
    details: dict[str, Any]


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    feature_group: str,
    seed: int,
    c: float = 1.0,
) -> TrainedCandidate:
    """Fit the required scaled Logistic Regression baseline on training rows only."""
    columns = feature_indices(feature_group)
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(C=c, solver="lbfgs", max_iter=1000, random_state=seed),
            ),
        ]
    )
    started = perf_counter()
    estimator.fit(X_train[:, columns], y_train)
    fit_seconds = perf_counter() - started
    probabilities = estimator.predict_proba(X_validation[:, columns])[:, 1]
    return TrainedCandidate(
        name=f"Logistic Regression | {feature_group}",
        feature_group=feature_group,
        estimator=estimator,
        validation_probabilities=probabilities,
        fit_seconds=fit_seconds,
        details={
            "C": c,
            "solver": "lbfgs",
            "max_iter": 1000,
            "validation": ranking_metrics(y_validation, probabilities),
        },
    )


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    feature_group: str,
    seed: int,
    parameters: dict[str, Any] | None = None,
) -> TrainedCandidate:
    """Fit XGBoost with validation-only early stopping and histogram trees."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is required for this candidate; install requirements-app.txt") from exc
    columns = feature_indices(feature_group)
    defaults: dict[str, Any] = {
        "n_estimators": 1200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "random_state": seed,
        "n_jobs": 1,
    }
    if parameters:
        defaults.update(parameters)
    estimator = XGBClassifier(**defaults)
    started = perf_counter()
    estimator.fit(
        X_train[:, columns],
        y_train,
        eval_set=[(X_validation[:, columns], y_validation)],
        verbose=False,
    )
    fit_seconds = perf_counter() - started
    probabilities = estimator.predict_proba(X_validation[:, columns])[:, 1]
    details = {**defaults, "best_iteration": getattr(estimator, "best_iteration", None)}
    details["validation"] = ranking_metrics(y_validation, probabilities)
    return TrainedCandidate(
        name=f"XGBoost | {feature_group}",
        feature_group=feature_group,
        estimator=estimator,
        validation_probabilities=probabilities,
        fit_seconds=fit_seconds,
        details=details,
    )


def train_compact_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    feature_group: str,
    seed: int,
    epochs: int = 100,
    patience: int = 12,
    batch_size: int = 512,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> TrainedCandidate:
    """Fit a compact MLP and restore the checkpoint selected by validation AUC."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the MLP; install requirements-app.txt") from exc

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    columns = feature_indices(feature_group)
    scaler = StandardScaler().fit(X_train[:, columns])
    train_x = scaler.transform(X_train[:, columns]).astype(np.float32)
    validation_x = scaler.transform(X_validation[:, columns]).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class CompactMLP(nn.Module):
        def __init__(self, width: int = 128) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(len(columns), width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(width, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(width, 1),
            )

        def forward(self, values):
            return self.network(values).squeeze(1)

    model = CompactMLP().to(device)
    optimizers = {
        "adamw": lambda: torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        "adagrad": lambda: torch.optim.Adagrad(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        ),
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"unknown MLP optimizer {optimizer_name!r}; choose from {sorted(optimizers)}")
    optimizer = optimizers[optimizer_name]()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_function = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_tensor = torch.from_numpy(validation_x).to(device)
    best_state, best_auc, best_epoch, stale = None, -np.inf, 0, 0
    history: list[dict[str, float]] = []
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_y)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_p = torch.sigmoid(model(validation_tensor)).cpu().numpy()
        validation_auc = ranking_metrics(y_validation, validation_p)["roc_auc"]
        history.append(
            {"epoch": epoch, "train_loss": total_loss / len(y_train), "validation_roc_auc": validation_auc}
        )
        if validation_auc > best_auc + 1e-5:
            best_auc, best_epoch, stale = validation_auc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    fit_seconds = perf_counter() - started
    if best_state is None:
        raise RuntimeError("MLP did not create a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(validation_tensor)).cpu().numpy()
    bundle = {"model": model, "scaler": scaler, "columns": columns, "device": str(device)}
    return TrainedCandidate(
        name=f"Compact MLP | {feature_group}",
        feature_group=feature_group,
        estimator=bundle,
        validation_probabilities=probabilities,
        fit_seconds=fit_seconds,
        details={
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler": "CosineAnnealingLR",
            "epochs_requested": epochs,
            "selected_epoch": best_epoch,
            "patience": patience,
            "batch_size": batch_size,
            "device": str(device),
            "history": history,
            "validation": ranking_metrics(y_validation, probabilities),
        },
    )


def train_best_compact_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    feature_group: str,
    seed: int,
    optimizer_names: tuple[str, ...] = ("adamw", "adagrad"),
) -> TrainedCandidate:
    """Run a bounded optimizer comparison and retain the validation winner."""
    if not optimizer_names:
        raise ValueError("at least one MLP optimizer must be supplied")
    trials = [
        train_compact_mlp(
            X_train,
            y_train,
            X_validation,
            y_validation,
            feature_group,
            seed,
            optimizer_name=optimizer,
        )
        for optimizer in optimizer_names
    ]
    selected = max(trials, key=lambda trial: trial.details["validation"]["roc_auc"])
    selected.details["optimizer_trials"] = [
        {
            "optimizer": trial.details["optimizer"],
            "validation_roc_auc": trial.details["validation"]["roc_auc"],
            "selected_epoch": trial.details["selected_epoch"],
        }
        for trial in trials
    ]
    return selected
