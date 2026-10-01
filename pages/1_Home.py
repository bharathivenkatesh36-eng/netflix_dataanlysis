"""Page 1 - Home dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import artifacts_exist, load_processed_data
from utils.styling import apply_netflix_theme, render_kpi_cards

st.set_page_config(page_title="Home | Netflix Dashboard", page_icon=":material/movie:", layout="wide")
apply_netflix_theme()

if not artifacts_exist():
    st.error(
        "Trained model artifacts were not found. Run `python train_models.py` from the project root, "
        "then reload the dashboard."
    )
    st.stop()

df = load_processed_data()
total_titles = len(df)
total_movies = int((df["type"] == "Movie").sum())
total_tv = int((df["type"] == "TV Show").sum())
total_countries = df["country"].str.split(", ").explode().nunique()

st.markdown(
    """
    <section class="home-hero">
        <div class="hero-copy">One focused workspace for understanding the Netflix catalog and the models built around it.</div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h3 class='section-head'>Catalog at a glance</h3>", unsafe_allow_html=True)
render_kpi_cards([
    {"label": "Total Titles", "value": f"{total_titles:,}"},
    {"label": "Movies", "value": f"{total_movies:,}"},
    {"label": "TV Shows", "value": f"{total_tv:,}"},
    {"label": "Countries", "value": f"{total_countries:,}"},
])

st.markdown("<h3 class='section-head'>Project summary</h3>", unsafe_allow_html=True)
st.markdown(
    """
    Explore the catalog from raw data quality through interactive EDA, leakage-safe supervised
    learning, clustering, content recommendations, and live Movie versus TV Show prediction.
    Every model is trained offline and loaded from the saved artifacts, keeping this dashboard
    fast, reproducible, and easy to inspect.
    """
)

st.markdown("<h3 class='section-head'>Quick access</h3>", unsafe_allow_html=True)
quick_access = [
    ("bar chart", "Filter the catalog and compare content, genre, geography, and time trends.", "pages/4_EDA.py", "bar_chart"),
    ("lightbulb", "See the strongest dataset and model findings in one concise view.", "pages/10_Insights.py", "lightbulb"),
    ("psychology", "Compare classifiers, clusters, metrics, and feature behavior.", "pages/5_Supervised_ML.py", "psychology"),
    ("auto awesome", "Find titles with similar metadata using TF-IDF and cosine similarity.", "pages/7_Recommendation_System.py", "auto_awesome"),
]
cols = st.columns(4)
for col, (title, description, path, icon) in zip(cols, quick_access):
    with col:
        st.markdown(
            f"""
            <div class="quick-card">
                <div class="quick-title">{title}</div>
                <div class="quick-description">{description}</div>
                <div class="quick-link"><span class="material-symbols-outlined">{icon}</span> Open workspace</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(path, label="Open workspace", icon=f":material/{icon}:")
