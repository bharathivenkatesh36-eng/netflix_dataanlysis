"""Page 10 - dynamic dataset, model, and technical insights."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import load_insights, missing_artifacts
from utils.styling import apply_netflix_theme, page_header

st.set_page_config(page_title="Insights | Netflix Dashboard", page_icon=":material/lightbulb:", layout="wide")
apply_netflix_theme()
page_header("Key Insights", "A decision-ready summary of the catalog and saved model results", "lightbulb")

missing = missing_artifacts(["insights.joblib"])
if missing:
    st.error(
        "The Key Insights artifact is unavailable. Missing: `insights.joblib`. "
        "Run `python train_models.py` from the project root to regenerate it. "
        "The rest of the dashboard remains available."
    )
    st.stop()

try:
    insights = load_insights()
except Exception as exc:
    st.error(
        "The Key Insights artifact could not be loaded. It may be incomplete or corrupted. "
        "Regenerate it with `python train_models.py`, then reload this page."
    )
    st.caption(f"Loader detail: {type(exc).__name__}")
    st.stop()


def insight_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="insight-stat">
            <div class="insight-stat-label">{html.escape(label)}</div>
            <div class="insight-stat-value">{html.escape(value)}</div>
            <div class="insight-stat-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("### Data insights")
data_cards = [
    ("Movie share", f"{insights['movie_pct']:.1%}", "of the catalog is Movies"),
    ("TV Show share", f"{1 - insights['movie_pct']:.1%}", "of the catalog is TV Shows"),
    ("Added since 2016", f"{insights['added_after_2016_pct']:.1%}", "of titles entered the catalog after 2016"),
    ("Leading country", str(insights["top_country"]), "highest title contribution"),
]
data_cols = st.columns(2)
for col, card in zip(data_cols, data_cards):
    with col:
        insight_card(*card)

st.markdown("### Catalog signals")
signal_cols = st.columns(2)
with signal_cols[0]:
    insight_card("Top genre", str(insights["top_genre"]), "most common genre tag")
with signal_cols[1]:
    insight_card("Catalog direction", "Recent-first", f"{insights['added_after_2016_pct']:.1%} added since 2016")

st.markdown("### Machine learning insights")
ml_cards = [
    ("Best model", str(insights["best_model_name"]), "highest saved F1 score"),
    ("F1 score", f"{insights['best_model_f1']:.2%}", "held-out classification result"),
    ("Optimal clusters", str(insights["optimal_k"]), f"silhouette score {insights['best_silhouette']:.3f}"),
]
ml_cols = st.columns(3)
for col, card in zip(ml_cols, ml_cards):
    with col:
        insight_card(*card)

st.markdown("### Technical highlights")
technical = [
    ("Leakage-safe classification", "Genre and duration signals that reveal the target were excluded before model training."),
    ("TF-IDF recommendations", "Sparse text vectors and on-demand cosine similarity keep recommendations responsive without storing an NxN matrix."),
    ("Offline artifacts", "Models and derived results are trained once, persisted with joblib, and loaded through cached utilities."),
]
technical_cols = st.columns(3)
for col, (title, description) in zip(technical_cols, technical):
    with col:
        st.markdown(
            f"<div class=\"netflix-card technical-card\"><h4>{html.escape(title)}</h4><p>{html.escape(description)}</p></div>",
            unsafe_allow_html=True,
        )

st.markdown("### Continue exploring")
links = st.columns(3)
with links[0]:
    st.page_link("pages/4_EDA.py", label="Exploratory Data Analysis", icon=":material/bar_chart:")
with links[1]:
    st.page_link("pages/8_Model_Performance.py", label="Model Performance", icon=":material/military_tech:")
with links[2]:
    st.page_link("pages/7_Recommendation_System.py", label="Recommendation System", icon=":material/auto_awesome:")
