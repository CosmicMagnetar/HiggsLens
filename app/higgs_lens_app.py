"""HiggsLens: an artifact-first research review workspace."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from higgslens.artifacts import matching_models, validate_prediction_frame
from higgslens.core import FEATURE_GROUPS, evaluate_frozen_candidate, ranking_metrics

st.set_page_config(page_title="HiggsLens | Research workspace", page_icon="⚛", layout="wide")

THEMES = {
    "Light": {
        "paper": "#f4f0e7",
        "surface": "#fffdf8",
        "ink": "#11213a",
        "muted": "#596574",
        "line": "#c6cdd2",
        "accent": "#155bb0",
        "signal": "#e65a47",
        "panel": "#e9e1d3",
        "sidebar": "#10233f",
        "sidebar_surface": "#17365e",
        "notice": "#fff2eb",
        "notice_ink": "#5d2d2a",
        "ok": "#d8f0e2",
        "shadow": "rgba(17, 33, 58, .10)",
    },
    "Dark": {
        "paper": "#0c1320",
        "surface": "#121e30",
        "ink": "#edf1ed",
        "muted": "#a8b5c3",
        "line": "#33465c",
        "accent": "#79b7ff",
        "signal": "#ff8b78",
        "panel": "#18283b",
        "sidebar": "#080e18",
        "sidebar_surface": "#152943",
        "notice": "#2a1d23",
        "notice_ink": "#ffd9d1",
        "ok": "#183f32",
        "shadow": "rgba(0, 0, 0, .35)",
    },
}


def read_csv(upload, path: Path):
    return pd.read_csv(upload) if upload is not None else (pd.read_csv(path) if path.exists() else None)


def read_json(upload, path: Path) -> dict:
    return (
        json.load(upload) if upload is not None else (json.loads(path.read_text()) if path.exists() else {})
    )


def metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"Model": name, **ranking_metrics(frame["label"], frame[name])}
        for name in frame.columns
        if name != "label"
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("roc_auc", ascending=False)
        .rename(
            columns={
                "roc_auc": "ROC-AUC",
                "average_precision": "Average precision",
                "accuracy_at_0_5": "Accuracy at 0.50",
                "brier_score": "Brier score",
                "log_loss": "Log loss",
                "prevalence": "Signal prevalence",
            }
        )
    )


def style_chart(ax, colors: dict[str, str]) -> None:
    ax.set_facecolor(colors["surface"])
    ax.figure.set_facecolor(colors["surface"])
    ax.tick_params(colors=colors["muted"])
    for spine in ax.spines.values():
        spine.set_color(colors["line"])
    ax.xaxis.label.set_color(colors["muted"])
    ax.yaxis.label.set_color(colors["muted"])
    ax.title.set_color(colors["ink"])
    ax.grid(alpha=0.16, color=colors["line"])


with st.sidebar:
    st.markdown("## HIGGSLENS")
    st.caption("Simulated-event benchmark review")

theme_name = st.session_state.get("theme_picker", "Dark")
colors = THEMES[theme_name]
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
:root {{ --paper:{colors["paper"]}; --surface:{colors["surface"]}; --ink:{colors["ink"]}; --muted:{colors["muted"]}; --line:{colors["line"]}; --accent:{colors["accent"]}; --signal:{colors["signal"]}; --panel:{colors["panel"]}; --sidebar:{colors["sidebar"]}; --sidebar-surface:{colors["sidebar_surface"]}; --notice:{colors["notice"]}; --notice-ink:{colors["notice_ink"]}; --ok:{colors["ok"]}; --shadow:{colors["shadow"]}; }}
.stApp {{ background:var(--paper); color:var(--ink); }} html, body, [class*="css"] {{ font-family:'DM Sans', sans-serif; }} [data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] {{ background:var(--sidebar); border-right:1px solid var(--line); }} [data-testid="stSidebar"] * {{ color:#edf1ed !important; }} [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ background:var(--sidebar-surface); border-color:#718aa7; }}
.hero {{ border-top:6px solid var(--accent); border-bottom:1px solid var(--line); padding:1.6rem 0 1.25rem; margin-bottom:1rem; position:relative; overflow:hidden; }} .hero::after {{ content:'01'; position:absolute; right:.1rem; bottom:-2.7rem; font-family:'Fraunces',serif; font-size:12rem; color:var(--panel); line-height:1; z-index:0; }} .hero>* {{ position:relative; z-index:1; }}
.eyebrow {{ font-family:'DM Mono',monospace; color:var(--accent); font-size:.74rem; letter-spacing:.13em; font-weight:500; }} .hero h1,.section-title {{ font-family:'Fraunces',serif; color:var(--ink); letter-spacing:-.055em; }} .hero h1 {{ font-size:clamp(3rem,6.5vw,6.25rem); line-height:.87; margin:.35rem 0 .7rem; max-width:56rem; }} .dek,.section-intro {{ color:var(--muted); max-width:50rem; line-height:1.62; }} .dek {{ font-size:1.05rem; }}
.nav-label {{ font-family:'DM Mono',monospace; color:var(--muted); font-size:.72rem; letter-spacing:.09em; text-transform:uppercase; margin:.65rem 0 -.25rem; }} [data-testid="stRadio"]>div {{ gap:.35rem; flex-wrap:wrap; }} [data-testid="stRadio"] label {{ border:1px solid var(--line); background:var(--surface); border-radius:0; padding:.6rem .8rem; min-height:42px; transition:transform .18s ease, background .18s ease; }} [data-testid="stRadio"] label:hover {{ transform:translateY(-2px); background:var(--panel); }} [data-testid="stRadio"] label p {{ color:var(--ink) !important; font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:.025em; }} [data-testid="stRadio"] input {{ accent-color:var(--accent); }}
[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--accent); padding:.9rem 1rem; min-height:106px; box-shadow:0 8px 24px var(--shadow); }} [data-testid="stMetricLabel"] {{ font-family:'DM Mono',monospace; font-size:.68rem; text-transform:uppercase; letter-spacing:.05em; }} [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * {{ color:var(--ink) !important; }} [data-testid="stMetricValue"] {{ font-family:'Fraunces',serif; }} [data-testid="stDataFrame"] {{ border:1px solid var(--line); }}
.scope {{ border-left:4px solid var(--signal); background:var(--notice); padding:.85rem 1rem; color:var(--notice-ink); margin:.7rem 0 1.3rem; line-height:1.55; }} .section-title {{ font-size:clamp(2rem,4vw,3.4rem); margin:1.2rem 0 .3rem; }} .section-intro {{ font-size:1rem; margin-bottom:1.2rem; }}
.info-card {{ background:var(--surface); border:1px solid var(--line); padding:1.1rem; min-height:170px; transition:transform .18s ease, box-shadow .18s ease; }} .info-card:hover {{ transform:translateY(-4px); box-shadow:0 14px 26px var(--shadow); }} .info-card h3 {{ color:var(--ink); font-family:'Fraunces',serif; margin:.05rem 0 .45rem; font-size:1.5rem; }} .info-card p {{ color:var(--muted); line-height:1.55; font-size:.92rem; }} .tag {{ display:inline-block; border:1px solid var(--line); color:var(--accent); font-family:'DM Mono',monospace; font-size:.68rem; padding:.2rem .4rem; margin:0 .25rem .4rem 0; }} .status-ok {{ display:inline-block; background:var(--ok); color:var(--ink); border:1px solid var(--line); padding:.3rem .7rem; font-family:'DM Mono',monospace; font-size:.7rem; letter-spacing:.04em; }} .timeline {{ border-left:1px solid var(--line); margin:1rem 0; padding-left:1.1rem; }} .timeline div {{ color:var(--muted); margin:0 0 1rem; line-height:1.5; }} .timeline strong {{ color:var(--ink); }}
.hero,[data-testid="stMetric"],.info-card {{ animation:settle .55s cubic-bezier(.2,.8,.2,1) both; }} @keyframes settle {{ from {{ opacity:0; transform:translateY(12px); }} to {{ opacity:1; transform:translateY(0); }} }} @media (max-width:700px) {{ .hero {{ padding-top:1.15rem; }} .hero h1 {{ font-size:3.15rem; }} .dek {{ font-size:1rem; }} }} @media (prefers-reduced-motion:reduce) {{ *,*::before,*::after {{ animation:none !important; transition:none !important; }} }}
</style>""",
    unsafe_allow_html=True,
)

artifact_dir = Path("higgs_project_outputs")
with st.sidebar:
    st.divider()
    st.markdown("### Load a run")
    st.caption(
        "Use artifacts from one completed benchmark. This app never trains or silently changes the final candidate."
    )
    val_upload = st.file_uploader("Validation predictions", type="csv")
    test_upload = st.file_uploader("Final-test predictions", type="csv")
    metadata_upload = st.file_uploader("Run metadata", type="json")
    st.divider()
    st.caption("Artifact review · simulated HIGGS only")

val_raw = read_csv(val_upload, artifact_dir / "val_predictions.csv")
test_raw = read_csv(test_upload, artifact_dir / "test_predictions.csv")
metadata = read_json(metadata_upload, artifact_dir / "run_metadata.json")
validation = test = None
load_error = None
if val_raw is not None:
    try:
        validation = validate_prediction_frame(val_raw, "validation predictions")
        if test_raw is not None:
            test = validate_prediction_frame(test_raw, "test predictions")
            matching_models(validation, test)
    except ValueError as exc:
        load_error = str(exc)

st.markdown(
    """<section class="hero"><div class="eyebrow">HIGGSLENS / RESEARCH WORKSPACE</div><h1>Evidence first.<br>Wonder, second.</h1><div class="dek">An approachable workspace for comparing simulated Higgs-event classifiers, exploring an operating threshold, and keeping development decisions separate from the final test result.</div></section>""",
    unsafe_allow_html=True,
)
navigation_header, theme_control = st.columns([4, 1])
with navigation_header:
    st.markdown("<div class='nav-label'>Navigate the workspace</div>", unsafe_allow_html=True)
with theme_control:
    st.radio("Appearance", ["Dark", "Light"], key="theme_picker", horizontal=True)
page = st.radio(
    "Navigate the workspace",
    ["01 / About", "02 / Develop", "03 / Frozen result", "04 / Feature lab", "05 / Research desk"],
    horizontal=True,
    label_visibility="collapsed",
)
if load_error:
    st.error(load_error)

if page == "01 / About":
    st.markdown(
        "<div class='section-title'>A benchmark with boundaries.</div><div class='section-intro'>HiggsLens makes the research workflow legible: what data was used, what a model was allowed to learn from, and what a score can—and cannot—mean.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='scope'><strong>Important scope.</strong> The UCI HIGGS dataset contains simulated Monte Carlo events. This is a benchmark-review product, not a detector control system, a clinical-style risk score, or evidence of a Higgs discovery.</div>",
        unsafe_allow_html=True,
    )
    cards = st.columns(3)
    content = [
        (
            "THE QUESTION",
            "Can a classifier separate patterns?",
            "We compare linear, tree-based, and neural models on the same declared event budget and feature groups.",
        ),
        (
            "THE GUARDRAIL",
            "Validation chooses. Test verifies.",
            "Models and operating thresholds are explored on validation predictions. A declared candidate is then evaluated once on the held-out test partition.",
        ),
        (
            "THE OUTPUT",
            "Useful evidence, not a trophy.",
            "ROC-AUC, average precision, calibration, background rejection, accuracy, timing, and configuration belong together.",
        ),
    ]
    for col, (tag, title, text) in zip(cards, content):
        col.markdown(
            f"<div class='info-card'><span class='tag'>{tag}</span><h3>{title}</h3><p>{text}</p></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='section-title'>How to use this workspace</div><div class='timeline'><div><strong>01 — Load a run.</strong> Use the sidebar to upload predictions and metadata, or launch with an artifact directory mounted by Docker.</div><div><strong>02 — Develop honestly.</strong> Compare validation candidates and choose an operating target in the Develop view.</div><div><strong>03 — Inspect once.</strong> Review the frozen candidate only after metadata names the candidate and target recall.</div><div><strong>04 — Learn the variables.</strong> Use Feature lab and Research desk to understand what the benchmark is measuring.</div></div>",
        unsafe_allow_html=True,
    )
    if validation is not None:
        row = metric_table(validation).iloc[0]
        overview = st.columns(4)
        overview[0].metric("Rows in loaded run", str(metadata.get("rows_loaded", len(validation))))
        overview[1].metric("Validation candidates", str(len(validation.columns) - 1))
        overview[2].metric("Top validation ROC-AUC", f"{row['ROC-AUC']:.4f}")
        overview[3].metric("Top candidate", str(row["Model"]).split("|")[0].strip())
    else:
        st.info(
            "No run is loaded yet. The product tour, feature guide, and research resources are available; load artifacts to unlock live benchmark views."
        )

elif page == "02 / Develop":
    st.markdown(
        "<div class='section-title'>Development console</div><div class='section-intro'>This screen deliberately uses validation predictions only. Tune an operating target here, then leave the final-test result untouched until a candidate is frozen.</div>",
        unsafe_allow_html=True,
    )
    if validation is None:
        st.warning("Load `val_predictions.csv` in the sidebar to open the development console.")
    else:
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
        models = [column for column in validation.columns if column != "label"]
        choice, operating = st.columns([1.25, 1])
        with choice:
            chosen_model = st.selectbox(
                "Validation candidate", models, help="Changing this never accesses final-test predictions."
            )
        with operating:
            target = st.slider(
                "Target signal efficiency",
                0.50,
                0.98,
                0.80,
                0.01,
                help="Choose on validation, record it, then apply it unchanged to the frozen test split.",
            )
        development = evaluate_frozen_candidate(
            validation["label"],
            validation[chosen_model],
            validation["label"],
            validation[chosen_model],
            target,
        )
        operating_metrics = development["test_metrics"]
        metrics = st.columns(5)
        for col, (name, value) in zip(
            metrics,
            [
                ("Validation threshold", f"{development['threshold_selection']['threshold']:.5f}"),
                ("Signal efficiency", f"{operating_metrics['signal_efficiency']:.1%}"),
                ("Background rejection", f"{operating_metrics['background_rejection']:.1%}"),
                ("Precision", f"{operating_metrics['precision']:.1%}"),
                ("Accuracy", f"{operating_metrics['accuracy']:.1%}"),
            ],
        ):
            col.metric(name, value)
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(6.4, 4.1))
            ax.hist(
                validation.loc[validation["label"] == 0, chosen_model],
                bins=24,
                density=True,
                alpha=0.62,
                color=colors["accent"],
                label="Background",
            )
            ax.hist(
                validation.loc[validation["label"] == 1, chosen_model],
                bins=24,
                density=True,
                alpha=0.55,
                color=colors["signal"],
                label="Signal",
            )
            ax.axvline(
                development["threshold_selection"]["threshold"],
                color=colors["ink"],
                linewidth=2,
                label="Selected threshold",
            )
            ax.set(xlabel="Model score", ylabel="Relative density", title="Validation score landscape")
            style_chart(ax, colors)
            ax.legend(frameon=False, labelcolor=colors["ink"])
            fig.tight_layout()
            st.pyplot(fig)
        with right:
            fpr, tpr, _ = roc_curve(validation["label"], validation[chosen_model])
            fig, ax = plt.subplots(figsize=(6.4, 4.1))
            ax.plot(
                fpr,
                tpr,
                color=colors["accent"],
                linewidth=2.5,
                label=f"ROC-AUC {roc_auc_score(validation['label'], validation[chosen_model]):.3f}",
            )
            ax.plot([0, 1], [0, 1], "--", color=colors["muted"], label="Random ranking")
            ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Validation ROC")
            style_chart(ax, colors)
            ax.legend(frameon=False, labelcolor=colors["ink"])
            fig.tight_layout()
            st.pyplot(fig)
        st.caption(
            "Accuracy at 0.50 is secondary context. In a selection problem, use the predeclared signal-efficiency target and report the matched background rejection."
        )
        st.download_button(
            "Download validation comparison",
            table.to_csv(index=False).encode("utf-8"),
            "validation_comparison.csv",
            "text/csv",
        )

elif page == "03 / Frozen result":
    st.markdown(
        "<div class='section-title'>Frozen final result</div><div class='section-intro'>This view stays intentionally quiet until the run metadata identifies a model and target chosen before looking at test labels. That is the line between evaluation and tuning.</div>",
        unsafe_allow_html=True,
    )
    freeze = metadata.get("frozen_candidate")
    if validation is None or test is None:
        st.info("Load both validation and final-test predictions to inspect a frozen evaluation.")
    elif not isinstance(freeze, dict) or not {"model", "target_recall"}.issubset(freeze):
        st.warning(
            "Final-test data is loaded, but it remains protected until `run_metadata.json` declares `frozen_candidate.model` and `frozen_candidate.target_recall`."
        )
    elif freeze["model"] not in validation.columns:
        st.error("The frozen model named in metadata is absent from the prediction files.")
    else:
        final = evaluate_frozen_candidate(
            validation["label"],
            validation[freeze["model"]],
            test["label"],
            test[freeze["model"]],
            float(freeze["target_recall"]),
        )
        result = final["test_metrics"]
        st.markdown("<span class='status-ok'>FROZEN CONFIGURATION LOADED</span>", unsafe_allow_html=True)
        st.caption(
            f"{freeze['model']} · threshold selected on validation for {float(freeze['target_recall']):.0%} target signal efficiency."
        )
        summary = st.columns(6)
        for col, (name, value) in zip(
            summary,
            [
                ("ROC-AUC", f"{result['roc_auc']:.4f}"),
                ("Average precision", f"{result['average_precision']:.4f}"),
                ("Signal efficiency", f"{result['signal_efficiency']:.1%}"),
                ("Background rejection", f"{result['background_rejection']:.1%}"),
                ("F1", f"{result['f1']:.3f}"),
                ("Accuracy", f"{result['accuracy']:.1%}"),
            ],
        ):
            col.metric(name, value)
        left, right = st.columns(2)
        with left:
            fpr, tpr, _ = roc_curve(test["label"], test[freeze["model"]])
            fig, ax = plt.subplots(figsize=(6.2, 4.3))
            ax.plot(
                fpr,
                tpr,
                color=colors["accent"],
                linewidth=2.3,
                label=f"ROC-AUC {roc_auc_score(test['label'], test[freeze['model']]):.3f}",
            )
            ax.plot([0, 1], [0, 1], "--", color=colors["muted"], label="Random ranking")
            ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Frozen-test ROC")
            style_chart(ax, colors)
            ax.legend(frameon=False, labelcolor=colors["ink"])
            fig.tight_layout()
            st.pyplot(fig)
        with right:
            fraction_positive, mean_predicted = calibration_curve(
                test["label"], test[freeze["model"]], n_bins=10, strategy="quantile"
            )
            fig, ax = plt.subplots(figsize=(6.2, 4.3))
            ax.plot(
                mean_predicted,
                fraction_positive,
                "o-",
                color=colors["signal"],
                linewidth=2,
                label="Frozen candidate",
            )
            ax.plot([0, 1], [0, 1], "--", color=colors["muted"], label="Perfect calibration")
            ax.set(
                xlabel="Mean predicted probability",
                ylabel="Observed signal fraction",
                title="Frozen-test reliability",
            )
            style_chart(ax, colors)
            ax.legend(frameon=False, labelcolor=colors["ink"])
            fig.tight_layout()
            st.pyplot(fig)
        st.caption(
            f"Brier score {brier_score_loss(test['label'], test[freeze['model']]):.4f}; average precision {average_precision_score(test['label'], test[freeze['model']]):.4f}. Treat calibration and discrimination as separate questions."
        )

elif page == "04 / Feature lab":
    st.markdown(
        "<div class='section-title'>Feature lab</div><div class='section-intro'>The benchmark asks whether models benefit from direct reconstructed quantities, physicist-derived combinations, or both. This is a comparison tool—not a claim that any one feature explains the Higgs boson.</div>",
        unsafe_allow_html=True,
    )
    selected_group = st.selectbox(
        "Explore a feature group",
        ["low_level", "high_level", "all_features"],
        format_func=lambda value: {
            "low_level": "Low-level · 21 inputs",
            "high_level": "High-level · 7 derived inputs",
            "all_features": "All features · 28 inputs",
        }[value],
    )
    descriptions = {
        "low_level": "Direct reconstructed lepton, missing-energy, and jet quantities. These make the model learn relationships from more elemental inputs.",
        "high_level": "Mass-like combinations designed from physics knowledge. They can make structured relationships more immediately available to a classical model.",
        "all_features": "The complete benchmark view: direct reconstruction variables plus derived mass-like quantities. Compare it at the same data budget and split as the smaller groups.",
    }
    st.markdown(
        f"<div class='scope'><strong>{selected_group.replace('_', ' ').title()}.</strong> {descriptions[selected_group]}</div>",
        unsafe_allow_html=True,
    )
    feature_names = FEATURE_GROUPS[selected_group]
    st.dataframe(
        pd.DataFrame(
            {
                "#": np.arange(1, len(feature_names) + 1),
                "Feature": feature_names,
                "Role": [
                    "direct reconstruction" if name in FEATURE_GROUPS["low_level"] else "derived combination"
                    for name in feature_names
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    cards = st.columns(3)
    for col, (tag, title, text) in zip(
        cards,
        [
            (
                "LINEAR",
                "Logistic Regression",
                "A transparent calibrated baseline after training-only scaling. Useful for asking what is gained by nonlinear structure.",
            ),
            (
                "TREES",
                "XGBoost",
                "Captures nonlinear interactions efficiently. Compare it under the same split and event budget; record early stopping and training time.",
            ),
            (
                "NEURAL",
                "Compact MLP",
                "Batch normalization, dropout, weight decay, and a bounded AdamW/AdaGrad validation comparison support a disciplined neural baseline.",
            ),
        ],
    ):
        col.markdown(
            f"<div class='info-card'><span class='tag'>{tag}</span><h3>{title}</h3><p>{text}</p></div>",
            unsafe_allow_html=True,
        )

else:
    st.markdown(
        "<div class='section-title'>Research desk</div><div class='section-intro'>A small reading room for understanding the physics context, the original benchmark, and the rules that keep model comparisons credible.</div>",
        unsafe_allow_html=True,
    )
    resources = st.columns(3)
    links = [
        (
            "PHYSICS",
            "CERN: the Higgs boson",
            "A plain-language starting point for what the Higgs field and boson are—and why a table of simulated features is not a discovery analysis.",
            "https://home.cern/science/physics/higgs-boson",
            "Read CERN's overview ↗",
        ),
        (
            "DATASET",
            "UCI HIGGS",
            "Dataset provenance, field description, and the original learning task. Record the source and exact row count for every run.",
            "https://archive.ics.uci.edu/dataset/280/higgs",
            "Open the UCI record ↗",
        ),
        (
            "PAPER",
            "Baldi et al. (2014)",
            "The benchmark paper behind the HIGGS task. Read it for context; do not copy its reported figures into a new experiment.",
            "https://doi.org/10.1038/ncomms5308",
            "View the paper ↗",
        ),
    ]
    for col, (tag, title, text, url, label) in zip(resources, links):
        col.markdown(
            f"<div class='info-card'><span class='tag'>{tag}</span><h3>{title}</h3><p>{text}</p><p><a href='{url}' target='_blank'>{label}</a></p></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='section-title'>What a credible run records</div><div class='section-intro'>A good result is reproducible enough for another student to challenge it constructively.</div>",
        unsafe_allow_html=True,
    )
    st.code(
        "source + row count + seed + split + feature group + model config + validation rule + frozen test metrics + environment",
        language="text",
    )
    if metadata:
        st.markdown("#### Loaded run record")
        split = metadata.get("split", {})
        st.dataframe(
            pd.DataFrame(
                [
                    {"Field": "Run kind", "Value": metadata.get("run_kind", "not recorded")},
                    {"Field": "Rows loaded", "Value": metadata.get("rows_loaded", "not recorded")},
                    {"Field": "Split protocol", "Value": split.get("protocol", "not recorded")},
                    {"Field": "Seed", "Value": split.get("seed", "not recorded")},
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        with st.expander("Full run metadata"):
            st.json(metadata)
    else:
        st.info("Load `run_metadata.json` to inspect the reproducibility record for a completed run.")
