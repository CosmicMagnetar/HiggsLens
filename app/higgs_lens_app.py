"""HiggsLens dashboard for reviewing a frozen classification benchmark run."""

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

st.set_page_config(page_title="HiggsLens | Benchmark Review", page_icon="⚛", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
:root { --ink:#12233b; --paper:#f5f2ea; --blue:#0a56a8; --red:#d64045; --line:#c9d0d4; --muted:#5d6874; }
.stApp { background: var(--paper); color: var(--ink); }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background:#10233f; }
[data-testid="stSidebar"] * { color:#f5f2ea !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:#17365e; border-color:#7891af; }
.masthead { border-top:7px solid var(--blue); border-bottom:1px solid var(--ink); padding:1.6rem 0 1.25rem; margin-bottom:1.1rem; }
.eyebrow { font-family:'DM Mono', monospace; color:var(--blue); font-size:.73rem; letter-spacing:.12em; text-transform:uppercase; font-weight:500; }
.masthead h1 { font-family:'Fraunces', serif; color:var(--ink); font-size:clamp(2.5rem, 6vw, 5.4rem); line-height:.9; margin:.25rem 0 .55rem; letter-spacing:-.045em; }
.dek { color:var(--muted); font-size:1.05rem; max-width:52rem; line-height:1.55; }
.stTabs [data-baseweb="tab-list"] { border-bottom:1px solid var(--line); gap:1.5rem; }
.stTabs [data-baseweb="tab"] { font-family:'DM Mono', monospace; font-size:.78rem; letter-spacing:.05em; padding: .8rem .1rem; }
.stTabs [aria-selected="true"] { color:var(--blue) !important; border-bottom-color:var(--blue) !important; }
[data-testid="stMetric"] { background:#fffdf8; border:1px solid var(--line); border-left:4px solid var(--blue); padding:.8rem 1rem; min-height:102px; }
[data-testid="stMetricLabel"] { font-family:'DM Mono', monospace; font-size:.68rem; text-transform:uppercase; letter-spacing:.04em; }
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] *, [data-testid="stMetricDelta"] * { color:var(--ink) !important; }
[data-testid="stMetricValue"] { color:var(--ink) !important; font-family:'Fraunces', serif; }
[data-testid="stDataFrame"] { border:1px solid var(--line); }
.method-note { border-left:4px solid var(--red); background:#fff5f0; padding:.75rem 1rem; color:#523135; margin:.5rem 0 1.2rem; }
.status-ok { display:inline-block; background:#dff0e6; color:#145a32; border-radius:99px; padding:.28rem .7rem; font-family:'DM Mono', monospace; font-size:.72rem; }
.masthead, [data-testid="stMetric"] { animation:settle .55s cubic-bezier(.2,.8,.2,1) both; }
[data-testid="stMetric"]:hover { transform:translateY(-3px); box-shadow:0 11px 22px rgba(18,35,59,.10); transition:transform .18s ease, box-shadow .18s ease; }
@keyframes settle { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@media (max-width: 700px) { .masthead h1 { font-size:3rem; } .dek { font-size:1rem; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation:none !important; transition:none !important; } }
</style>
""",
    unsafe_allow_html=True,
)


def read_csv(upload, path: Path):
    """Read an uploaded artifact or its local mounted equivalent."""
    return pd.read_csv(upload) if upload is not None else (pd.read_csv(path) if path.exists() else None)


def read_json(upload, path: Path) -> dict:
    """Read optional run metadata without treating a missing file as a frozen run."""
    if upload is not None:
        return json.load(upload)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a readable, validation-only model comparison table."""
    rows = [
        {"Model": name, **ranking_metrics(frame["label"], frame[name])}
        for name in frame.columns
        if name != "label"
    ]
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    return table.rename(
        columns={
            "roc_auc": "ROC-AUC",
            "average_precision": "Average precision",
            "accuracy_at_0_5": "Accuracy at 0.50",
            "brier_score": "Brier score",
            "log_loss": "Log loss",
            "prevalence": "Signal prevalence",
        }
    )


artifact_dir = Path("higgs_project_outputs")
with st.sidebar:
    st.markdown("### Run artifacts")
    st.caption("Upload files from one completed benchmark run, or mount them at `higgs_project_outputs/`.")
    val_upload = st.file_uploader("Validation predictions", type="csv")
    test_upload = st.file_uploader("Final-test predictions", type="csv")
    metadata_upload = st.file_uploader("Run metadata", type="json")
    st.divider()
    st.caption("The dashboard does not train models or choose a final candidate.")

st.markdown(
    """
<section class="masthead">
  <div class="eyebrow">HIGGSLENS / EXPERIMENT REVIEW</div>
  <h1>See the evidence.<br>Keep the boundary.</h1>
  <div class="dek">A review workspace for simulated HIGGS classification runs. Explore model development on validation predictions, then inspect the prespecified final result without turning the test split into a tuning tool.</div>
</section>
""",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='method-note'><strong>Scope.</strong> UCI HIGGS is simulated Monte Carlo data. This interface reviews benchmark artifacts; it is not particle-detector software, a discovery system, or a real-event probability service.</div>",
    unsafe_allow_html=True,
)

val_raw = read_csv(val_upload, artifact_dir / "val_predictions.csv")
test_raw = read_csv(test_upload, artifact_dir / "test_predictions.csv")
metadata = read_json(metadata_upload, artifact_dir / "run_metadata.json")
if val_raw is None:
    st.info(
        "Start by loading validation predictions. The development workspace will open without final-test data."
    )
    st.stop()
try:
    validation = validate_prediction_frame(val_raw, "validation predictions")
except ValueError as exc:
    st.error(str(exc))
    st.stop()

models = [column for column in validation.columns if column != "label"]
rows_loaded = metadata.get("rows_loaded", "not recorded")
split = metadata.get("split", {})
top_model = metric_table(validation).iloc[0]
overview = st.columns(4)
overview[0].metric("Rows in run", str(rows_loaded))
overview[1].metric("Validation models", str(len(models)))
overview[2].metric("Top validation ROC-AUC", f"{top_model['ROC-AUC']:.4f}")
overview[3].metric("Top validation model", str(top_model["Model"]).split("|")[0].strip())

development_tab, final_tab, run_tab = st.tabs(["01 / Development", "02 / Frozen result", "03 / Run record"])
with development_tab:
    st.markdown("#### Validation workspace")
    st.caption(
        "These views use validation labels and scores only. Changing the operating target below does not inspect the final test partition."
    )
    table = metric_table(validation)
    st.dataframe(
        table.style.format(
            {
                "ROC-AUC": "{:.4f}",
                "Average precision": "{:.4f}",
                "Accuracy at 0.50": "{:.2%}",
                "Brier score": "{:.4f}",
                "Log loss": "{:.4f}",
                "Signal prevalence": "{:.2%}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    choice, operating = st.columns([1.2, 1])
    with choice:
        chosen_model = st.selectbox("Inspect a validation candidate", models)
    with operating:
        target = st.slider("Target signal efficiency", 0.50, 0.98, 0.80, 0.01)
    development = evaluate_frozen_candidate(
        validation["label"], validation[chosen_model], validation["label"], validation[chosen_model], target
    )
    operating_metrics = development["test_metrics"]
    numbers = st.columns(4)
    numbers[0].metric("Validation threshold", f"{development['threshold_selection']['threshold']:.5f}")
    numbers[1].metric("Signal efficiency", f"{operating_metrics['signal_efficiency']:.1%}")
    numbers[2].metric("Background rejection", f"{operating_metrics['background_rejection']:.1%}")
    numbers[3].metric("Precision", f"{operating_metrics['precision']:.1%}")
    st.caption(
        "Accuracy at 0.50 is secondary context. The cards above use the validation-selected operating threshold."
    )
    st.download_button(
        "Download validation comparison",
        table.to_csv(index=False).encode("utf-8"),
        "validation_comparison.csv",
        "text/csv",
    )

with final_tab:
    st.markdown("#### Frozen final evaluation")
    if test_raw is None:
        st.info("No final-test file has been loaded. This is a valid development-only review state.")
    else:
        try:
            test = validate_prediction_frame(test_raw, "test predictions")
            matching_models(validation, test)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        freeze = metadata.get("frozen_candidate")
        if not isinstance(freeze, dict) or not {"model", "target_recall"}.issubset(freeze):
            st.warning(
                "Final-test data is intentionally hidden until metadata declares the frozen model and target recall."
            )
        elif freeze["model"] not in models:
            st.error("The frozen model named in metadata is absent from the prediction files.")
        else:
            final = evaluate_frozen_candidate(
                validation["label"],
                validation[freeze["model"]],
                test["label"],
                test[freeze["model"]],
                float(freeze["target_recall"]),
            )
            st.markdown("<span class='status-ok'>FROZEN CONFIGURATION LOADED</span>", unsafe_allow_html=True)
            st.caption(
                f"{freeze['model']} - threshold chosen on validation for {float(freeze['target_recall']):.0%} signal efficiency."
            )
            result = final["test_metrics"]
            summary = st.columns(6)
            summary[0].metric("ROC-AUC", f"{result['roc_auc']:.4f}")
            summary[1].metric("Average precision", f"{result['average_precision']:.4f}")
            summary[2].metric("Signal efficiency", f"{result['signal_efficiency']:.1%}")
            summary[3].metric("Background rejection", f"{result['background_rejection']:.1%}")
            summary[4].metric("F1", f"{result['f1']:.3f}")
            summary[5].metric("Accuracy", f"{result['accuracy']:.1%}")
            left, right = st.columns(2)
            yt, pt = test["label"], test[freeze["model"]]
            with left:
                fpr, tpr, _ = roc_curve(yt, pt)
                fig, ax = plt.subplots(figsize=(6.2, 4.5))
                ax.plot(
                    fpr, tpr, color="#0a56a8", linewidth=2.3, label=f"ROC-AUC {roc_auc_score(yt, pt):.3f}"
                )
                ax.plot([0, 1], [0, 1], "--", color="#8b949e", label="Random ranking")
                ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Final-test ROC")
                ax.legend(frameon=False)
                fig.tight_layout()
                st.pyplot(fig)
            with right:
                fraction_positive, mean_predicted = calibration_curve(yt, pt, n_bins=10, strategy="quantile")
                fig, ax = plt.subplots(figsize=(6.2, 4.5))
                ax.plot(
                    mean_predicted,
                    fraction_positive,
                    "o-",
                    color="#d64045",
                    linewidth=2,
                    label="Frozen candidate",
                )
                ax.plot([0, 1], [0, 1], "--", color="#8b949e", label="Perfect calibration")
                ax.set(
                    xlabel="Mean predicted probability",
                    ylabel="Observed signal fraction",
                    title="Final-test reliability",
                )
                ax.legend(frameon=False)
                fig.tight_layout()
                st.pyplot(fig)
            st.caption(
                f"Brier score {brier_score_loss(yt, pt):.4f}; average precision {average_precision_score(yt, pt):.4f}. Read Brier with the reliability chart because it includes discrimination as well as calibration effects."
            )

with run_tab:
    st.markdown("#### Reproducibility record")
    if metadata:
        context = pd.DataFrame(
            [
                {"Field": "Run kind", "Value": metadata.get("run_kind", "not recorded")},
                {"Field": "Rows loaded", "Value": rows_loaded},
                {"Field": "Split protocol", "Value": split.get("protocol", "not recorded")},
                {"Field": "Seed", "Value": split.get("seed", "not recorded")},
            ]
        )
        st.dataframe(context, hide_index=True, use_container_width=True)
        with st.expander("Full run metadata"):
            st.json(metadata)
    else:
        st.info(
            "No metadata file was supplied. The validation workspace remains available, but final evaluation stays protected."
        )
