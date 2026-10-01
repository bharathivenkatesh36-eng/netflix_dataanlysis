"""Page 11 — Live Prediction.

Lets a user describe a brand-new, unseen title and get a Movie/TV Show
prediction from the already-trained best model — with input validation,
a confidence indicator, a rule-based (non-SHAP) explanation, prediction
history, and CSV/PDF export. Nothing on this page retrains anything; every
prediction uses only the artifacts saved by train_models.py.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import (
    load_classification_models,
    load_explanation_stats,
    load_inference_defaults,
    load_metrics,
    load_model_metadata,
    missing_artifacts,
)
from utils.helper import (
    build_rule_based_explanation,
    confidence_level,
    prediction_record_to_csv_bytes,
    prediction_record_to_pdf_bytes,
    validate_live_prediction_inputs,
)
from utils.preprocessing import CLASSIFIER_NUMERIC_FEATURES, build_live_input_features
from utils.styling import apply_netflix_theme, insight_box, netflix_plotly_layout, page_header, render_kpi_cards

st.set_page_config(page_title="Live Prediction | Netflix Dashboard", page_icon=":material/target:", layout="wide")
apply_netflix_theme()
page_header("Live Prediction", "Predict Movie vs. TV Show for a brand-new, unseen title", "target")

# ---------------------------------------------------------------------------
# Graceful artifact loading — item 14: missing models/artifacts
# ---------------------------------------------------------------------------
REQUIRED_ARTIFACTS = [
    "processed_data.pkl", "classification_model.joblib", "metrics.joblib",
    "inference_defaults.joblib", "model_metadata.joblib", "explanation_stats.joblib",
]
missing = missing_artifacts(REQUIRED_ARTIFACTS)
if missing:
    st.error(
        "This page can't run yet — the following trained artifacts are missing from `models/`:\n\n"
        + "\n".join(f"- `{m}`" for m in missing)
        + "\n\nRun **`python train_models.py`** once from the project root to generate them. "
        "This page never trains models itself — it only loads what that script produces."
    )
    st.stop()

try:
    metrics = load_metrics()
    models = load_classification_models()
    defaults = load_inference_defaults()
    model_metadata = load_model_metadata()
    explanation_stats = load_explanation_stats()
except Exception as exc:
    st.error(
        "Something went wrong loading the trained artifacts. They may be corrupted or were "
        f"trained with a different library version. Try re-running `python train_models.py`.\n\n"
        f"Details: {exc}"
    )
    st.stop()

best_model_name: str = model_metadata["best_model_name"]
if best_model_name not in models:
    st.error(
        f"The best model on record (**{best_model_name}**) isn't present in the saved "
        "classification models file. Try re-running `python train_models.py`."
    )
    st.stop()
best_pipeline = models[best_model_name]

CURRENT_YEAR = datetime.now().year

# ---------------------------------------------------------------------------
# Model Information card — item 2 (fully dynamic, nothing hardcoded)
# ---------------------------------------------------------------------------
st.markdown("### Model Information")
render_kpi_cards([
    {"label": "Best Model", "value": best_model_name},
    {"label": "Model Version", "value": model_metadata["model_version"]},
    {"label": "Training Date", "value": model_metadata["trained_at"].split("T")[0]},
    {"label": "Cross Validation F1", "value": f"{model_metadata['best_cv_f1_mean']:.3f} ± {model_metadata['best_cv_f1_std']:.3f}"},
    {"label": "Best F1 Score", "value": f"{model_metadata['best_f1_score']:.3f}"},
])
st.caption(
    f"Trained on {model_metadata['training_rows']:,} titles · scikit-learn {model_metadata['sklearn_version']} · "
    f"{model_metadata['cv_folds']}-fold cross-validation"
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Pipeline visualization — item 3
# ---------------------------------------------------------------------------
with st.expander("How This Page Works — Machine Learning Pipeline", expanded=False):
    steps = ["User Input", "Feature Engineering", "Preprocessing", "Best Model", "Prediction", "Confidence Score"]
    arrow = '<span style="color:#7A7A7A; margin:0 0.5rem;">↓</span>'
    steps_html = ""
    for i, s in enumerate(steps):
        steps_html += (
            f'<div style="background:#1F1F1F; border:1px solid #2a2a2a; border-left:3px solid #E50914; '
            f'border-radius:6px; padding:0.5rem 1rem; color:white; font-weight:600; font-size:0.9rem; '
            f'text-align:center; margin-bottom:0.15rem;">{i+1}. {s}</div>'
        )
        if i < len(steps) - 1:
            steps_html += f'<div style="text-align:center;">{arrow}</div>'
    st.markdown(f'<div style="max-width:340px; margin:0 auto;">{steps_html}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        This page loads the already-trained **{best_model_name}** pipeline (the best model by
        F1 Score — see Model Performance) with `joblib` and **never retrains it**. Your inputs
        are validated, then turned into the same feature set the model was trained on
        (`utils.preprocessing.build_live_input_features`), the fitted `ColumnTransformer` inside
        the pipeline applies the exact same scaling/encoding/TF-IDF used during training, and the
        model predicts directly.

        **Note on Genre:** this form doesn't ask for a Genre, on purpose — genre (`listed_in`)
        is excluded from the classifier entirely to prevent data leakage (see Data Preprocessing
        page), so it isn't part of what the model looks at. A dataset-wide average genre count
        is used internally to fill that numeric slot.
        """
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Session state — form defaults + prediction history
# ---------------------------------------------------------------------------
FORM_DEFAULTS = {
    "lp_release_year": defaults["max_release_year"],
    "lp_rating": defaults["rating_options"][0] if defaults["rating_options"] else "TV-MA",
    "lp_country": "",
    "lp_director": "",
    "lp_cast": "",
    "lp_description": "",
}
for key, default_value in FORM_DEFAULTS.items():
    st.session_state.setdefault(key, default_value)
st.session_state.setdefault("lp_history", [])

# ---------------------------------------------------------------------------
# Input form — item 4: validation, item 12: reset button
# ---------------------------------------------------------------------------
st.markdown("### 📝 Describe the Title")
st.caption("Fields marked with * are required.")

with st.form("live_prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        release_year = st.number_input(
            "Release Year *", min_value=1888, max_value=CURRENT_YEAR, key="lp_release_year",
        )
        rating = st.selectbox("Rating *", options=defaults["rating_options"], key="lp_rating")
        country = st.text_input(
            "Country *", key="lp_country",
            placeholder="e.g. United States, Canada",
            help="One or more countries, comma-separated.",
        )
    with c2:
        director = st.text_input(
            "Director *", key="lp_director",
            placeholder="e.g. Jane Doe",
            help="One or more directors, comma-separated.",
        )
        cast = st.text_input(
            "Cast *", key="lp_cast",
            placeholder="e.g. Actor A, Actor B, Actor C",
            help="One or more cast members, comma-separated.",
        )
    description = st.text_area(
        "Description * (minimum 20 characters)", key="lp_description", height=100,
        placeholder="A short synopsis of the title...",
    )

    b1, b2 = st.columns(2)
    with b1:
        predict_clicked = st.form_submit_button("Predict", type="primary", width="stretch")
    with b2:
        reset_clicked = st.form_submit_button("Reset Form", width="stretch")

if reset_clicked:
    for key, default_value in FORM_DEFAULTS.items():
        st.session_state[key] = default_value
    st.rerun()

# ---------------------------------------------------------------------------
# Prediction — item 4 (validate first), item 6 (staged spinners + timing),
# item 14 (graceful error handling)
# ---------------------------------------------------------------------------
if predict_clicked:
    validation_errors = validate_live_prediction_inputs(
        release_year=int(release_year), rating=rating, country=country, director=director,
        cast=cast, description=description, valid_ratings=defaults["rating_options"],
        max_year=CURRENT_YEAR,
    )

    if validation_errors:
        st.error("Please fix the following before predicting:\n\n" + "\n".join(f"- {e}" for e in validation_errors))
        st.stop()

    try:
        timings: dict[str, float] = {}

        with st.spinner("Loading model..."):
            t0 = time.perf_counter()
            pipeline = best_pipeline  # already loaded via cached loader — this just times the handoff
            timings["Load Model"] = time.perf_counter() - t0

        with st.spinner("Engineering features..."):
            t0 = time.perf_counter()
            input_df = build_live_input_features(
                release_year=int(release_year), rating=rating, country=country, director=director,
                cast=cast, description=description, reference_year=defaults["reference_year"],
                default_number_of_genres=defaults["default_number_of_genres"],
            )
            timings["Engineer Features"] = time.perf_counter() - t0

        with st.spinner("Running prediction..."):
            t0 = time.perf_counter()
            prediction = pipeline.predict(input_df)[0]
            predicted_label = "TV Show" if prediction == 1 else "Movie"
            confidence = None
            proba_movie = proba_tv = None
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(input_df)[0]
                proba_movie, proba_tv = float(proba[0]), float(proba[1])
                confidence = max(proba_movie, proba_tv)
            timings["Prediction"] = time.perf_counter() - t0

        with st.spinner("Generating explanation..."):
            t0 = time.perf_counter()
            explanation_lines = build_rule_based_explanation(
                input_row=input_df.iloc[0].to_dict(), explanation_stats=explanation_stats,
                predicted_label=predicted_label, numeric_features=CLASSIFIER_NUMERIC_FEATURES,
            )
            timings["Explanation"] = time.perf_counter() - t0

        total_inference_time = sum(timings.values())

    except Exception as exc:
        st.error(
            "Something went wrong while generating the prediction. This usually means one of "
            f"the inputs couldn't be processed. Details: {exc}"
        )
        st.stop()

    prediction_timestamp = datetime.now()
    emoji, conf_label = confidence_level(confidence)

    # -----------------------------------------------------------------
    # Result card — item 7 & item 8 (confidence indicator)
    # -----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Prediction Result")

    badge_color = "#E50914" if predicted_label == "TV Show" else "#F5576C"
    confidence_text = f"{confidence:.1%}" if confidence is not None else "N/A"

    st.markdown(
        f"""
        <div class="netflix-card" style="border-top:4px solid {badge_color}; border-left:none;">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:1.5rem;">
                <div>
                    <div style="color:#B3B3B3; font-size:0.82rem; letter-spacing:1px;">PREDICTED TYPE</div>
                    <div style="color:{badge_color}; font-size:2rem; font-weight:800;">{predicted_label}</div>
                </div>
                <div>
                    <div style="color:#B3B3B3; font-size:0.82rem; letter-spacing:1px;">CONFIDENCE</div>
                    <div style="color:white; font-size:2rem; font-weight:800;">{confidence_text}</div>
                    <div style="color:#B3B3B3; font-size:0.85rem;">{emoji} {conf_label}</div>
                </div>
                <div>
                    <div style="color:#B3B3B3; font-size:0.82rem; letter-spacing:1px;">MODEL USED</div>
                    <div style="color:white; font-size:1.3rem; font-weight:700; margin-top:0.4rem;">{best_model_name}</div>
                </div>
                <div>
                    <div style="color:#B3B3B3; font-size:0.82rem; letter-spacing:1px;">PREDICTED AT</div>
                    <div style="color:white; font-size:1rem; font-weight:600; margin-top:0.4rem;">{prediction_timestamp.strftime("%Y-%m-%d %H:%M:%S")}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1.2])
    with c1:
        if proba_movie is not None:
            st.markdown("#### Prediction Probability")
            fig = go.Figure(go.Bar(
                x=[proba_movie, proba_tv], y=["Movie", "TV Show"], orientation="h",
                marker_color=["#E50914" if predicted_label == "Movie" else "#7A7A7A",
                              "#E50914" if predicted_label == "TV Show" else "#7A7A7A"],
                text=[f"{proba_movie:.1%}", f"{proba_tv:.1%}"], textposition="auto",
            ))
            fig.update_layout(xaxis_range=[0, 1], showlegend=False)
            st.plotly_chart(netflix_plotly_layout(fig, title="Class Probabilities", height=260), width="stretch")

        st.markdown("#### Timing")
        timing_lines = "".join(f"<li>{name}: {seconds*1000:.2f} ms</li>" for name, seconds in timings.items())
        st.markdown(
            f"""
            <div class="netflix-card">
                <ul style="color:#B3B3B3; font-size:0.88rem; margin:0; padding-left:1.2rem;">
                    {timing_lines}
                </ul>
                <p style="color:white; font-weight:700; margin-top:0.6rem; margin-bottom:0;">
                    Total Inference Time: {total_inference_time*1000:.2f} ms
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("#### 📝 Input Summary")
        st.markdown(
            f"""
            <div class="netflix-card">
                <p style="color:#B3B3B3; font-size:0.9rem; line-height:1.9;">
                    <b style="color:white;">Release Year:</b> {int(release_year)}<br>
                    <b style="color:white;">Rating:</b> {rating}<br>
                    <b style="color:white;">Country:</b> {country}<br>
                    <b style="color:white;">Director:</b> {director}<br>
                    <b style="color:white;">Cast:</b> {cast}<br>
                    <b style="color:white;">Description:</b> {description.strip()}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Engineered Features"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), width="stretch")

    # -----------------------------------------------------------------
    # Explanation — item 9 (rule-based, not fabricated SHAP)
    # -----------------------------------------------------------------
    st.markdown("#### Explanation")
    st.caption("Rule-based, computed by comparing this title's feature values to each class's historical average — not SHAP or any per-instance model attribution.")
    for line in explanation_lines:
        insight_box(line)

    # -----------------------------------------------------------------
    # Save to history — item 10
    # -----------------------------------------------------------------
    record = {
        "Time": prediction_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "Predicted Type": predicted_label,
        "Confidence": confidence_text,
        "Model Used": best_model_name,
        "Release Year": int(release_year),
        "Rating": rating,
        "Country": country,
        "Director": director,
        "Cast": cast,
        "Description": description.strip(),
    }
    st.session_state.lp_history.append(record)
    st.session_state.lp_history = st.session_state.lp_history[-50:]  # bound memory usage

    # -----------------------------------------------------------------
    # Download report — item 11
    # -----------------------------------------------------------------
    st.markdown("#### 📥 Download This Prediction")
    d1, d2 = st.columns(2)
    record_key = prediction_timestamp.strftime("%Y%m%d_%H%M%S")
    with d1:
        st.download_button(
            "⬇️ Download as CSV", data=prediction_record_to_csv_bytes(record),
            file_name=f"netflix_prediction_{record_key}.csv", mime="text/csv",
            width="stretch", key=f"csv_{record_key}",
        )
    with d2:
        st.download_button(
            "⬇️ Download as PDF", data=prediction_record_to_pdf_bytes(record),
            file_name=f"netflix_prediction_{record_key}.pdf", mime="application/pdf",
            width="stretch", key=f"pdf_{record_key}",
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Prediction history — item 10
# ---------------------------------------------------------------------------
st.markdown("### Prediction History (this session)")
if st.session_state.lp_history:
    history_df = pd.DataFrame(st.session_state.lp_history)[["Time", "Predicted Type", "Confidence", "Model Used"]]
    st.dataframe(history_df.iloc[::-1], width="stretch", hide_index=True)
    if st.button("🗑️ Clear History"):
        st.session_state.lp_history = []
        st.rerun()
else:
    st.info("No predictions yet this session — fill in the form above and click Predict.")
