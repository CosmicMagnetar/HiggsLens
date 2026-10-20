"""Artifact-only dashboard; development exploration is validation-only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from higgslens.artifacts import matching_models, validate_prediction_frame
from higgslens.core import evaluate_frozen_candidate, ranking_metrics

st.set_page_config(page_title="HiggsLens", page_icon="⚛️", layout="wide")
st.title("HiggsLens")
st.caption("Reproducible HIGGS benchmark artifact viewer")
st.warning(
    "Educational research prototype using simulated Monte Carlo data; not a particle-discovery or detector service."
)


def read_csv(upload, path: Path):
    return pd.read_csv(upload) if upload is not None else (pd.read_csv(path) if path.exists() else None)


def read_json(upload, path: Path) -> dict:
    if upload is not None:
        return json.load(upload)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


artifact_dir = Path("higgs_project_outputs")
with st.sidebar:
    st.header("Load frozen-run artifacts")
    val_upload = st.file_uploader("val_predictions.csv", type="csv")
    test_upload = st.file_uploader("test_predictions.csv", type="csv")
    metadata_upload = st.file_uploader("run_metadata.json", type="json")

val_raw = read_csv(val_upload, artifact_dir / "val_predictions.csv")
test_raw = read_csv(test_upload, artifact_dir / "test_predictions.csv")
metadata = read_json(metadata_upload, artifact_dir / "run_metadata.json")
if val_raw is None:
    st.info(
        "Upload validation predictions to explore model rankings and operating points without exposing test feedback."
    )
    st.stop()
try:
    validation = validate_prediction_frame(val_raw, "validation predictions")
except ValueError as exc:
    st.error(str(exc))
    st.stop()
models = [column for column in validation.columns if column != "label"]

st.subheader("Development view — validation only")
st.caption("Changing the target here never reads test labels or produces test metrics.")
candidate_rows = [
    {"Model": model, **ranking_metrics(validation["label"], validation[model])} for model in models
]
st.dataframe(
    pd.DataFrame(candidate_rows).sort_values("roc_auc", ascending=False),
    hide_index=True,
    use_container_width=True,
)
chosen_model = st.selectbox("Candidate for validation exploration", models)
target = st.slider("Validation target signal efficiency", 0.50, 0.98, 0.80, 0.01)
development = evaluate_frozen_candidate(
    validation["label"], validation[chosen_model], validation["label"], validation[chosen_model], target
)
st.metric("Validation-derived threshold", f"{development['threshold_selection']['threshold']:.5f}")

if test_raw is None:
    st.info("No final evaluation shown. A final-test view requires test predictions and freeze metadata.")
    st.stop()
try:
    test = validate_prediction_frame(test_raw, "test predictions")
    matching_models(validation, test)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

freeze = metadata.get("frozen_candidate")
if not isinstance(freeze, dict) or not {"model", "target_recall"}.issubset(freeze):
    st.warning(
        "Test predictions are loaded but hidden: run_metadata.json lacks frozen_candidate.model and frozen_candidate.target_recall. Final evaluation must be declared before opening this view."
    )
    st.stop()
if freeze["model"] not in models:
    st.error("The frozen model is absent from the matching prediction files.")
    st.stop()

st.divider()
st.subheader("Final evaluation — frozen candidate only")
st.caption("The model and target come from saved metadata, not dashboard controls.")
final = evaluate_frozen_candidate(
    validation["label"],
    validation[freeze["model"]],
    test["label"],
    test[freeze["model"]],
    float(freeze["target_recall"]),
)
st.json({"frozen_candidate": freeze, **final["threshold_selection"]})
st.dataframe(pd.DataFrame([final["test_metrics"]]), hide_index=True, use_container_width=True)

yt, pt = test["label"], test[freeze["model"]]
left, right = st.columns(2)
with left:
    fpr, tpr, _ = roc_curve(yt, pt)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(yt, pt):.3f}")
    ax.plot([0, 1], [0, 1], "--", label="random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Frozen final-test ROC")
    ax.legend()
    st.pyplot(fig)
with right:
    fraction_positive, mean_predicted = calibration_curve(yt, pt, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots()
    ax.plot(mean_predicted, fraction_positive, "o-", label=freeze["model"])
    ax.plot([0, 1], [0, 1], "--", label="perfect calibration")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed signal fraction",
        title="Final-test reliability diagram",
    )
    ax.legend()
    st.pyplot(fig)
st.caption(
    f"Final AP={average_precision_score(yt, pt):.4f}; Brier={brier_score_loss(yt, pt):.4f}. Brier reflects discrimination and calibration, not calibration alone."
)
