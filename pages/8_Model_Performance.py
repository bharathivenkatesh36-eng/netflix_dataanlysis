"""Page 8 — Model Performance."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helper import format_seconds
from utils.loader import artifacts_exist, load_metrics
from utils.styling import apply_netflix_theme, insight_box, netflix_plotly_layout, page_header, render_kpi_cards

st.set_page_config(page_title="Model Performance | Netflix Dashboard", page_icon=":material/military_tech:", layout="wide")
apply_netflix_theme()
page_header("Model Performance", "Final comparison, best model, and feature importance", "military_tech")

if not artifacts_exist():
    st.error("Model artifacts not found. Run `python train_models.py` first.")
    st.stop()

metrics = load_metrics()
results_df: pd.DataFrame = metrics["results_df"]
best_model_name: str = metrics["best_model_name"]
confusion_matrices = metrics["confusion_matrices"]
feature_importances = metrics["feature_importances"]

best_row = results_df[results_df["Model"] == best_model_name].iloc[0]

st.markdown(
    f"""
    <div class="netflix-card" style="text-align:center; border-left: none; border-top: 4px solid #E50914;">
        <div style="color:#B3B3B3; font-size:0.9rem; letter-spacing:1px;">BEST SUPERVISED MODEL</div>
        <div style="color:#E50914; font-size:2.2rem; font-weight:800; margin:0.2rem 0;">{best_model_name}</div>
        <div style="color:#B3B3B3;">Selected automatically by highest F1 Score</div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_kpi_cards([
    {"label": "Accuracy", "value": f"{best_row['Accuracy']:.2%}"},
    {"label": "Precision", "value": f"{best_row['Precision']:.2%}"},
    {"label": "Recall", "value": f"{best_row['Recall']:.2%}"},
    {"label": "F1 Score", "value": f"{best_row['F1 Score']:.2%}"},
    {"label": "Training Time", "value": format_seconds(best_row["Training Time (s)"])},
])

st.markdown("---")

# ---------------------------------------------------------------------------
# Comparison charts
# ---------------------------------------------------------------------------
st.markdown("### Metric-by-Metric Comparison")
metric_tabs = st.tabs(["Accuracy", "Precision", "Recall", "F1 Score"])
for tab, metric_name in zip(metric_tabs, ["Accuracy", "Precision", "Recall", "F1 Score"]):
    with tab:
        sorted_df = results_df.sort_values(metric_name, ascending=True)
        colors = ["#E50914" if m == best_model_name else "#7A7A7A" for m in sorted_df["Model"]]
        fig = px.bar(sorted_df, x=metric_name, y="Model", orientation="h", title=f"{metric_name} by Model")
        fig.update_traces(marker_color=colors)
        st.plotly_chart(netflix_plotly_layout(fig, height=380), width="stretch")

st.markdown("---")

# ---------------------------------------------------------------------------
# Training / prediction time
# ---------------------------------------------------------------------------
st.markdown("### Training Time vs. Prediction Time")
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(
        results_df.sort_values("Training Time (s)"), x="Training Time (s)", y="Model", orientation="h",
        title="Training Time (seconds)",
    )
    st.plotly_chart(netflix_plotly_layout(fig, height=380), width="stretch")
with c2:
    fig = px.bar(
        results_df.sort_values("Prediction Time (s/sample)"), x="Prediction Time (s/sample)", y="Model", orientation="h",
        title="Prediction Time (seconds/sample)",
    )
    st.plotly_chart(netflix_plotly_layout(fig, height=380), width="stretch")
insight_box("Faster models are attractive when latency matters; the best F1 score doesn't always come from the cheapest model to run.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Confusion matrix + feature importance for best model
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"### Confusion Matrix — {best_model_name}")
    cm = confusion_matrices[best_model_name]
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Reds",
        x=["Predicted: Movie", "Predicted: TV Show"], y=["Actual: Movie", "Actual: TV Show"],
    )
    st.plotly_chart(netflix_plotly_layout(fig, height=420), width="stretch")
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    insight_box(f"{tp + tn:,} of {tp+tn+fp+fn:,} test titles were classified correctly ({(tp+tn)/(tp+tn+fp+fn):.1%} accuracy).")

with c2:
    st.markdown("### Feature Importance")
    if "Random Forest" in feature_importances and not feature_importances["Random Forest"].empty:
        fi = feature_importances["Random Forest"].reset_index()
        fi.columns = ["feature", "importance"]
        fig = px.bar(
            fi.sort_values("importance"), x="importance", y="feature", orientation="h",
            title="Top 15 Features — Random Forest",
        )
        st.plotly_chart(netflix_plotly_layout(fig, height=420), width="stretch")
        insight_box("Feature importance is only available for tree-based models (Random Forest here). Higher bars mean the feature contributed more to the model's split decisions.")
    else:
        st.info("Feature importance is only computed for tree-based models.")

st.markdown("---")
st.markdown("### Full Comparison Table")
st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
        "F1 Score": "{:.4f}", "ROC AUC": "{:.4f}", "CV F1 Mean": "{:.4f}", "CV F1 Std": "{:.4f}",
        "Training Time (s)": "{:.3f}", "Prediction Time (s/sample)": "{:.6f}",
    }).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"], color="#831010"),
    width="stretch", hide_index=True,
)
