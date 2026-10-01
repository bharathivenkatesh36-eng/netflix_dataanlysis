"""Page 5 — Supervised Machine Learning."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helper import format_seconds
from utils.loader import artifacts_exist, load_classification_models, load_metrics
from utils.styling import apply_netflix_theme, insight_box, netflix_plotly_layout, page_header, render_kpi_cards

st.set_page_config(page_title="Supervised ML | Netflix Dashboard", page_icon=":material/psychology:", layout="wide")
apply_netflix_theme()
page_header("Supervised Machine Learning", "Predicting Movie vs. TV Show — leakage-safe, cross-validated", "psychology")

if not artifacts_exist():
    st.error("Model artifacts not found. Run `python train_models.py` first.")
    st.stop()

models = load_classification_models()
metrics = load_metrics()
results_df: pd.DataFrame = metrics["results_df"]
best_model_name: str = metrics["best_model_name"]
confusion_matrices = metrics["confusion_matrices"]
class_reports = metrics["classification_reports"]
roc_curves = metrics["roc_curves"]
tuning = metrics["hyperparameter_tuning"]
cv_folds = metrics["cv_folds"]

# ---------------------------------------------------------------------------
# Training process & leakage explanation
# ---------------------------------------------------------------------------
with st.expander("Training Process, Pipeline & Leakage Prevention", expanded=False):
    st.markdown(
        f"""
        **Task:** Binary classification — predict whether a title is a `Movie` (0) or
        `TV Show` (1) purely from its metadata.

        **Features used:** `{'`, `'.join(metrics['feature_columns'])}`

        **Deliberately excluded (data leakage):** `{'`, `'.join(metrics['excluded_features'])}`.
        Netflix's own genre tags (`listed_in`) literally contain the word "TV" for ~96% of TV
        Show rows and never for Movies — including it as a feature was verified to push every
        model to ~100% accuracy for the wrong reason. `duration` and its two derived columns
        are each populated for only one class, so their presence/absence alone would give the
        label away. Removing all four makes this a genuinely harder, honest classification
        task (see Data Preprocessing page for the full explanation).

        **Pipeline:** every model is a single scikit-learn `Pipeline` combining a
        `ColumnTransformer` (StandardScaler for numeric features, OneHotEncoder for
        categoricals, TfidfVectorizer for description text) with the classifier — so
        preprocessing is fit only on the training split, with no leakage into the test set.

        **Split:** 80% train / 20% test, stratified by class.

        **Cross-validation:** {cv_folds}-fold stratified CV (scored on F1) run on the training
        split for every model, in addition to the held-out test-set metrics below — this
        checks that performance is stable across different train/validation partitions, not
        a lucky split.

        **Hyperparameter tuning (demo):** a small `GridSearchCV` was run on {tuning['model']}
        over `{tuning['param_grid']}`, selecting **{tuning['best_params']}**
        (CV F1 = {tuning['best_cv_f1']:.4f}).

        All models were trained once in `train_models.py` and saved with `joblib` — this page
        only loads and displays the results.
        """
    )

# ---------------------------------------------------------------------------
# Comparison table with best model highlighted
# ---------------------------------------------------------------------------
st.markdown("### Model Comparison")

display_df = results_df.copy()
display_df["Training Time"] = display_df["Training Time (s)"].apply(format_seconds)
display_df["Prediction Time"] = display_df["Prediction Time (s/sample)"].apply(lambda x: f"{x*1000:.3f} ms")
display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "CV F1 Mean", "Training Time", "Prediction Time"]

def highlight_best(row):
    color = "background-color: #831010; color: white; font-weight: 700;" if row["Model"] == best_model_name else ""
    return [color] * len(row)

styled = display_df[display_cols].style.apply(highlight_best, axis=1).format({
    "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
    "F1 Score": "{:.4f}", "ROC AUC": "{:.4f}", "CV F1 Mean": "{:.4f}",
})
st.dataframe(styled, width="stretch", hide_index=True)
st.success(f"🥇 **Best model (by F1 Score): {best_model_name}**")

fig = px.bar(
    results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
                     var_name="Metric", value_name="Score"),
    x="Model", y="Score", color="Metric", barmode="group", title="Model Comparison Across Metrics",
)
fig.update_layout(yaxis_range=[0, 1.05])
st.plotly_chart(netflix_plotly_layout(fig), width="stretch")

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(
        results_df.sort_values("Training Time (s)"), x="Training Time (s)", y="Model", orientation="h",
        title="Training Time by Model (seconds)",
    )
    st.plotly_chart(netflix_plotly_layout(fig, height=380), width="stretch")
with c2:
    fig = px.bar(
        results_df.sort_values("Prediction Time (s/sample)"), x="Prediction Time (s/sample)", y="Model", orientation="h",
        title="Prediction Time by Model (seconds/sample)",
    )
    st.plotly_chart(netflix_plotly_layout(fig, height=380), width="stretch")
insight_box("Simpler models (Logistic Regression, KNN) train and predict fastest; ensemble methods like Random Forest cost more compute for a modest accuracy gain.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Per-model detail
# ---------------------------------------------------------------------------
st.markdown("### Per-Model Detail")
model_names = results_df["Model"].tolist()
selected_model = st.selectbox("Choose a model to inspect", options=model_names, index=model_names.index(best_model_name))

row = results_df[results_df["Model"] == selected_model].iloc[0]
render_kpi_cards([
    {"label": "Accuracy", "value": f"{row['Accuracy']:.2%}"},
    {"label": "Precision", "value": f"{row['Precision']:.2%}"},
    {"label": "Recall", "value": f"{row['Recall']:.2%}"},
    {"label": "F1 Score", "value": f"{row['F1 Score']:.2%}"},
    {"label": "CV F1 (mean ± std)", "value": f"{row['CV F1 Mean']:.3f} ± {row['CV F1 Std']:.3f}"},
])

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrices[selected_model]
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Reds",
        x=["Predicted: Movie", "Predicted: TV Show"], y=["Actual: Movie", "Actual: TV Show"],
    )
    st.plotly_chart(netflix_plotly_layout(fig, title=f"Confusion Matrix — {selected_model}", height=420), width="stretch")

with c2:
    st.markdown("#### Classification Report")
    report_df = pd.DataFrame(class_reports[selected_model]).T.round(3)
    st.dataframe(report_df, width="stretch", height=280)

if selected_model in roc_curves:
    st.markdown("#### ROC Curve")
    rc = roc_curves[selected_model]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rc["fpr"], y=rc["tpr"], mode="lines", name=f"{selected_model} (AUC = {rc['auc']:.3f})", line=dict(color="#E50914", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(color="#7A7A7A", dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(netflix_plotly_layout(fig, title=f"ROC Curve — {selected_model}"), width="stretch")
    insight_box(f"An AUC of {rc['auc']:.3f} means the model ranks a random TV Show above a random Movie {rc['auc']:.1%} of the time.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Pointer to the dedicated Live Prediction page
# ---------------------------------------------------------------------------
st.markdown("### Try a Prediction")
st.markdown(
    """
    Predicting on a brand-new, unseen title has its own dedicated page — with full feature
    engineering from raw inputs, input validation, a confidence indicator, a rule-based
    explanation, prediction history, and CSV/PDF export.
    """
)
if st.button("Open Live Prediction", type="primary"):
    st.switch_page("pages/11_Live_Prediction.py")
